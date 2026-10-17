use aerie_falcon_native::{Falcon1024, Falcon512, FalconParams};
use aerie_falcon_statement::{AggregationStatement, ProductionError, MAX_PRODUCTION_BATCH_SIZE};
use aerie_falcon_witness::AggregationWitness;

#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

fn allocated_on_rejection<T>(parse: impl FnOnce() -> Result<T, ProductionError>) -> u64 {
    let before = dhat::HeapStats::get().total_bytes;
    let result = parse();
    let allocated = dhat::HeapStats::get().total_bytes - before;
    assert!(matches!(
        result,
        Err(ProductionError::MalformedWire(ref message))
            if message == "truncated aggregate input"
    ));
    allocated
}

fn check<P: FalconParams>(count: u32) -> [u64; 2] {
    let mut header = [1, 0, 0, 0, 0, 0];
    header[2..].copy_from_slice(&count.to_le_bytes());
    let statement = allocated_on_rejection(|| AggregationStatement::<P>::parse_canonical(&header));
    let witness = allocated_on_rejection(|| AggregationWitness::<P>::parse_canonical(&header));
    eprintln!(
        "{} count={count}: statement allocated {statement} B; witness allocated {witness} B",
        P::NAME
    );
    [statement, witness]
}

// Keep one test in this binary: dhat tracks allocations process-wide.
#[test]
fn truncated_tables_reject_without_count_proportional_allocation() {
    let _profiler = dhat::Profiler::builder().testing().build();
    let mut largest = 0;
    for count in [1, 2, MAX_PRODUCTION_BATCH_SIZE as u32] {
        for allocated in check::<Falcon512>(count)
            .into_iter()
            .chain(check::<Falcon1024>(count))
        {
            largest = largest.max(allocated);
        }
    }
    // Allow small error/reporting overhead, but no record or signature table.
    assert!(largest <= 256, "parser allocated {largest} bytes");
}
