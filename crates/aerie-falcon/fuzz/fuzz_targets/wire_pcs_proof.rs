#![no_main]
use aerie_falcon_fuzz::check_wire_pcs_proof;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    check_wire_pcs_proof(data);
});
