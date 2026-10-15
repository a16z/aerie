//! [`Bytes`] — a byte field a candidate can supply *by reference*.
//!
//! The statements used by `completeness` and `statement-binding`, and the
//! envelope used by `verifier-totality`, are kilobytes of proof material, so a
//! byte field accepts either form:
//!
//! - inline — `[1, 2, 3]`, the literal bytes;
//! - by path — `{"file": "pair.bin"}` or just `"pair.bin"`, read from disk at
//!   deserialization time.
//!
//! Reads are capped by [`MAX_FILE_BYTES`] so a stray path cannot exhaust
//! memory; each invariant then applies its own, tighter cap.

use std::fs::{File, OpenOptions};
use std::io::Read as _;
use std::ops::Deref;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt as _;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Upper bound on a file-backed byte field, before the invariant's own cap.
pub const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// A byte string supplied inline or by file path. Derefs to the bytes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Bytes(pub Vec<u8>);

impl Deref for Bytes {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<[u8]> for Bytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<Vec<u8>> for Bytes {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

/// Serialized as the literal bytes, so a round-trip never depends on a file that
/// may be gone by the time the input is replayed from a log.
impl Serialize for Bytes {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

/// The accepted spellings of a byte field.
#[derive(Deserialize)]
#[serde(untagged)]
enum Repr {
    /// The literal bytes.
    Inline(Vec<u8>),
    /// `{"file": "/path"}`.
    File { file: PathBuf },
    /// A bare path string.
    Path(String),
}

impl<'de> Deserialize<'de> for Bytes {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Repr::deserialize(deserializer)? {
            Repr::Inline(bytes) => Ok(Self(bytes)),
            Repr::File { file } => read_capped(&file).map(Self).map_err(D::Error::custom),
            Repr::Path(path) => read_capped(&PathBuf::from(path))
                .map(Self)
                .map_err(D::Error::custom),
        }
    }
}

/// Open `path` without letting a FIFO block, require a regular file, and read no
/// more than one byte past [`MAX_FILE_BYTES`]. The extra byte detects files that
/// grow after their metadata is inspected as well as virtual regular files
/// whose metadata reports a zero length.
fn read_capped(path: &Path) -> Result<Vec<u8>, String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NONBLOCK);
    let file = options
        .open(path)
        .map_err(|error| format!("could not open `{}`: {error}", path.display()))?;
    read_opened_file(file, path)
}

fn read_opened_file(file: File, path: &Path) -> Result<Vec<u8>, String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("could not stat `{}`: {error}", path.display()))?;
    if !metadata.file_type().is_file() {
        return Err(format!("`{}` is not a regular file", path.display()));
    }
    if metadata.len() > MAX_FILE_BYTES {
        return Err(format!(
            "`{}` is {} bytes, past the {MAX_FILE_BYTES}-byte cap",
            path.display(),
            metadata.len()
        ));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("could not read `{}`: {error}", path.display()))?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err(format!(
            "`{}` grew past the {MAX_FILE_BYTES}-byte cap while being read",
            path.display()
        ));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::{Bytes, MAX_FILE_BYTES};

    #[test]
    fn accepts_inline_bytes() {
        let parsed: Bytes = serde_json::from_str("[1, 2, 3]").expect("inline");
        assert_eq!(*parsed, vec![1, 2, 3]);
    }

    #[test]
    fn accepts_a_file_object_and_a_bare_path() {
        let path = std::env::temp_dir().join(format!("aerie-bytes-{}.bin", std::process::id()));
        std::fs::write(&path, [7_u8, 8, 9]).expect("write");
        let display = path.display().to_string();

        let object: Bytes =
            serde_json::from_value(serde_json::json!({ "file": display })).expect("file object");
        assert_eq!(*object, vec![7, 8, 9]);

        let bare: Bytes = serde_json::from_value(serde_json::json!(display)).expect("bare path");
        assert_eq!(*bare, vec![7, 8, 9]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_missing_file_is_a_deserialization_error() {
        let missing = serde_json::json!({ "file": "/nonexistent/aerie-bytes-test.bin" });
        let error = serde_json::from_value::<Bytes>(missing).expect_err("should fail");
        assert!(error.to_string().contains("could not open"), "{error}");
    }

    #[test]
    fn rejects_an_oversized_regular_file_before_reading_it() {
        let path =
            std::env::temp_dir().join(format!("aerie-bytes-oversized-{}.bin", std::process::id()));
        let file = std::fs::File::create(&path).expect("create sparse oversized file");
        file.set_len(MAX_FILE_BYTES + 1)
            .expect("extend sparse oversized file");

        let spelling = serde_json::json!({ "file": path.display().to_string() });
        let error = serde_json::from_value::<Bytes>(spelling).expect_err("should reject cap");
        assert!(error.to_string().contains("past the"), "{error}");
        let _ = std::fs::remove_file(path);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_a_non_regular_device_without_reading_it() {
        let spelling = serde_json::json!({ "file": "/dev/zero" });
        let error = serde_json::from_value::<Bytes>(spelling).expect_err("device is not a file");
        assert!(error.to_string().contains("not a regular file"), "{error}");
    }

    #[test]
    fn serializes_back_to_literal_bytes() {
        let encoded = serde_json::to_string(&Bytes(vec![1, 2])).expect("encode");
        assert_eq!(encoded, "[1,2]");
    }
}
