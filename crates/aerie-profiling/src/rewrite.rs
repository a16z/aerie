//! Post-run trace rewrite: convert the monitor's `counters.*` instant
//! events into Chrome/Perfetto counter events (`"ph": "C"`), which
//! Perfetto renders as continuous memory/CPU tracks. Port of jolt's
//! `scripts/postprocess_trace.py` into the guard that flushes the trace
//! (`specs/techniques/profiling.md` section 6): the artifact is complete
//! the moment the process exits, with no external script.

use std::fs;
use std::io;
use std::path::Path;

use serde_json::{Map, Value};

const COUNTER_PREFIX: &str = "counters.";

/// Rewrites the trace at `path` in place, atomically (temp file + rename,
/// so a crash mid-rewrite cannot destroy a long run's trace).
pub fn rewrite_counters(path: &Path) -> io::Result<()> {
    let raw = fs::read_to_string(path)?;
    let trace: Value =
        serde_json::from_str(&raw).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let rewritten = rewrite_value(trace);
    let serialized = serde_json::to_string(&rewritten)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serialized)?;
    fs::rename(&tmp, path)
}

/// tracing-chrome writes a bare event array; the `traceEvents` object form
/// is accepted too so the rewrite stays valid on externally produced files.
fn rewrite_value(trace: Value) -> Value {
    match trace {
        Value::Array(events) => Value::Array(rewrite_events(events)),
        Value::Object(mut top) => {
            match top.remove("traceEvents") {
                Some(Value::Array(events)) => {
                    top.insert(
                        "traceEvents".to_owned(),
                        Value::Array(rewrite_events(events)),
                    );
                }
                // A non-array value is preserved untouched, never dropped.
                Some(other) => {
                    top.insert("traceEvents".to_owned(), other);
                }
                None => {}
            }
            Value::Object(top)
        }
        other => other,
    }
}

fn rewrite_events(events: Vec<Value>) -> Vec<Value> {
    let mut out = Vec::with_capacity(events.len());
    for event in events {
        match counter_events(&event) {
            Some(counters) => out.extend(counters),
            None => out.push(event),
        }
    }
    out
}

/// Returns the replacement counter events if `event` carries `counters.*`
/// args, or `None` to keep the event untouched.
fn counter_events(event: &Value) -> Option<Vec<Value>> {
    let obj = event.as_object()?;
    let args = obj.get("args")?.as_object()?;
    let counters: Vec<(&String, &Value)> = args
        .iter()
        .filter(|(key, _)| key.starts_with(COUNTER_PREFIX))
        .collect();
    if counters.is_empty() {
        return None;
    }
    let mut replacements = Vec::with_capacity(counters.len());
    for (key, value) in counters {
        let name = &key[COUNTER_PREFIX.len()..];
        let mut counter = Map::new();
        counter.insert("ph".to_owned(), Value::String("C".to_owned()));
        counter.insert("name".to_owned(), Value::String(name.to_owned()));
        for copied in ["ts", "pid", "tid"] {
            if let Some(v) = obj.get(copied) {
                counter.insert(copied.to_owned(), v.clone());
            }
        }
        let mut counter_args = Map::new();
        counter_args.insert(name.to_owned(), value.clone());
        counter.insert("args".to_owned(), Value::Object(counter_args));
        replacements.push(Value::Object(counter));
    }
    Some(replacements)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn monitor_event() -> Value {
        json!({
            "ph": "i", "ts": 12.5, "pid": 7, "tid": 3,
            "name": "monitor.rs:42",
            "args": {
                "counters.memory_gib": 1.25,
                "counters.cpu_percent": 640.0
            }
        })
    }

    #[test]
    fn counter_events_are_rewritten_and_original_dropped() {
        let span_event = json!({"ph": "B", "ts": 1.0, "pid": 7, "tid": 1, "name": "fp64.setup"});
        let trace = Value::Array(vec![span_event.clone(), monitor_event()]);
        let Value::Array(out) = rewrite_value(trace) else {
            panic!("array in, array out");
        };
        assert_eq!(out[0], span_event);
        assert_eq!(out.len(), 3, "one span event + two counter events");
        for counter in &out[1..] {
            assert_eq!(counter["ph"], "C");
            assert_eq!(counter["ts"], 12.5);
            assert_eq!(counter["pid"], 7);
            assert_eq!(counter["tid"], 3);
        }
        let names: Vec<&str> = out[1..]
            .iter()
            .map(|c| c["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"memory_gib") && names.contains(&"cpu_percent"));
        let mem = out[1..].iter().find(|c| c["name"] == "memory_gib").unwrap();
        assert_eq!(mem["args"]["memory_gib"], 1.25);
    }

    #[test]
    fn non_counter_events_pass_through_untouched() {
        let plain = json!([
            {"ph": "B", "name": "x", "args": {"batch": 8}},
            {"ph": "E", "name": "x"}
        ]);
        assert_eq!(rewrite_value(plain.clone()), plain);
    }

    #[test]
    fn trace_events_object_form_is_handled() {
        let trace = json!({"traceEvents": [monitor_event()], "displayTimeUnit": "ms"});
        let out = rewrite_value(trace);
        assert_eq!(out["displayTimeUnit"], "ms");
        assert_eq!(out["traceEvents"].as_array().unwrap().len(), 2);
        assert_eq!(out["traceEvents"][0]["ph"], "C");
    }

    #[test]
    fn rewrite_is_atomic_on_disk() {
        let dir = std::env::temp_dir().join(format!("aerie-rewrite-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trace.json");
        std::fs::write(
            &path,
            serde_json::to_string(&json!([monitor_event()])).unwrap(),
        )
        .unwrap();
        rewrite_counters(&path).unwrap();
        let rewritten: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(rewritten.as_array().unwrap().len(), 2);
        assert!(
            !dir.join("trace.json.tmp").exists(),
            "temp file renamed away"
        );

        // A malformed trace must leave the original bytes in place.
        std::fs::write(&path, "not json").unwrap();
        assert!(rewrite_counters(&path).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "not json");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
