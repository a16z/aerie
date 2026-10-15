use aerie::falcon::{
    prove_aggregate, verify_aggregate, AggregationError as Error, AggregationParams,
    CanonicalPublicKeyBytes, CanonicalSignatureBytes, Falcon1024, Falcon512, FalconAggregateSetup,
    MalformedProof, RejectionStage, Stability, StatementMismatch, PROTOCOL_STABILITY, WIRE_VERSION,
};
use aerie::falcon::{AggregationRecord, AggregationStatement, AggregationWitness};
use fn_dsa::{
    sign_key_size, signature_size, vrfy_key_size, CryptoRng, DomainContext, KeyPairGenerator,
    KeyPairGenerator1024, KeyPairGenerator512, RngCore, RngError, SigningKey, SigningKey1024,
    SigningKey512, VerifyingKey, VerifyingKey1024, VerifyingKey512, HASH_ID_RAW, SHAKE256,
};

const RECORDS: usize = 4;
const EMPTY_CONTEXT: DomainContext = DomainContext(b"");

// A seeded SHAKE stream makes test keys reproducible; these keys are public fixtures.
struct FixtureRng(SHAKE256);

impl RngCore for FixtureRng {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0; 4];
        self.fill_bytes(&mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0; 8];
        self.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.extract(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for FixtureRng {}

trait ReferenceParams: AggregationParams {
    type Generator: KeyPairGenerator;
    type Signer: SigningKey;
    type Verifier: VerifyingKey;
}

impl ReferenceParams for Falcon512 {
    type Generator = KeyPairGenerator512;
    type Signer = SigningKey512;
    type Verifier = VerifyingKey512;
}

impl ReferenceParams for Falcon1024 {
    type Generator = KeyPairGenerator1024;
    type Signer = SigningKey1024;
    type Verifier = VerifyingKey1024;
}

fn honest_batch_and_rejections<P: ReferenceParams>(other_proof: Option<&[u8]>) -> Vec<u8> {
    let mut stream = SHAKE256::new();
    stream.inject(b"aerie/falcon/facade/v1");
    stream.inject(P::NAME.as_bytes());
    stream.flip();
    let mut rng = FixtureRng(stream);
    let mut records = Vec::with_capacity(RECORDS);
    let mut signatures = Vec::with_capacity(RECORDS);
    for index in 0..RECORDS {
        let mut signing_key = vec![0; sign_key_size(P::LOGN)];
        let mut public_key = vec![0; vrfy_key_size(P::LOGN)];
        P::Generator::default().keygen(P::LOGN, &mut rng, &mut signing_key, &mut public_key);
        let message = format!("{} facade record {index}", P::NAME).into_bytes();
        let mut signer = P::Signer::decode(&signing_key).expect("reference signing key");
        let mut signature = vec![0; signature_size(P::LOGN)];
        signer.sign(
            &mut rng,
            &EMPTY_CONTEXT,
            &HASH_ID_RAW,
            &message,
            &mut signature,
        );
        assert!(P::Verifier::decode(&public_key)
            .expect("reference verifying key")
            .verify(&signature, &EMPTY_CONTEXT, &HASH_ID_RAW, &message));
        let public_key = CanonicalPublicKeyBytes::<P>::parse(&public_key).unwrap();
        let signature = CanonicalSignatureBytes::<P>::parse(&signature).unwrap();
        records.push(AggregationRecord::new(public_key, message).unwrap());
        signatures.push(signature);
    }
    let statement = AggregationStatement::new(records).unwrap();
    let witness = AggregationWitness::new(signatures.clone()).unwrap();
    let setup = FalconAggregateSetup::<P>::new(RECORDS).unwrap();
    let proof = prove_aggregate(&setup, &statement, &witness).unwrap();
    let verified: Result<(), Error> = verify_aggregate(setup.verifier(), &statement, &proof);
    verified.unwrap();

    assert_statement_mismatches(&setup, &statement, &proof, other_proof);

    let mut altered_records = statement.to_records();
    let mut altered_message = altered_records[0].message().to_vec();
    altered_message[0] ^= 1;
    altered_records[0] =
        AggregationRecord::new(altered_records[0].public_key().clone(), altered_message).unwrap();
    let altered_statement = AggregationStatement::new(altered_records).unwrap();
    let error = verify_aggregate(setup.verifier(), &altered_statement, &proof).unwrap_err();
    assert!(
        matches!(
            error,
            Error::ProofRejected { stage } if stage == RejectionStage::Grinding
        ),
        "{error:?}"
    );

    let mut swapped_records = statement.to_records();
    assert_ne!(
        swapped_records[0].public_key(),
        swapped_records[1].public_key()
    );
    swapped_records[0] = AggregationRecord::new(
        swapped_records[1].public_key().clone(),
        swapped_records[0].message().to_vec(),
    )
    .unwrap();
    let swapped_statement = AggregationStatement::new(swapped_records).unwrap();
    let error = verify_aggregate(setup.verifier(), &swapped_statement, &proof).unwrap_err();
    assert!(
        matches!(
            error,
            Error::ProofRejected { stage } if stage == RejectionStage::Grinding
        ),
        "{error:?}"
    );

    let error =
        verify_aggregate(setup.verifier(), &statement, &proof[..proof.len() - 1]).unwrap_err();
    assert!(
        matches!(error, Error::MalformedProof(MalformedProof::Truncated)),
        "{error:?}"
    );

    let mut flipped_proof = proof.clone();
    *flipped_proof.last_mut().unwrap() ^= 1;
    let error = verify_aggregate(setup.verifier(), &statement, &flipped_proof).unwrap_err();
    assert!(
        matches!(
            error,
            Error::ProofRejected { stage } if stage == RejectionStage::PcsOpening
        ),
        "{error:?}"
    );

    // The replacement signature is canonical but authenticates another record.
    signatures[0] = signatures[1].clone();
    assert!(
        !P::Verifier::decode(statement.public_keys()[0].as_bytes().as_ref())
            .unwrap()
            .verify(
                signatures[0].as_bytes().as_ref(),
                &EMPTY_CONTEXT,
                &HASH_ID_RAW,
                &statement.messages()[0],
            )
    );
    let invalid_witness = AggregationWitness::new(signatures).unwrap();
    let error = prove_aggregate(&setup, &statement, &invalid_witness).unwrap_err();
    assert!(matches!(error, Error::InvalidStatement(_)), "{error:?}");
    proof
}

fn assert_statement_mismatches<P: ReferenceParams>(
    setup: &FalconAggregateSetup<P>,
    statement: &AggregationStatement<P>,
    proof: &[u8],
    other_proof: Option<&[u8]>,
) {
    let short_statement =
        AggregationStatement::new(statement.to_records()[..RECORDS - 1].to_vec()).unwrap();
    let error = verify_aggregate(setup.verifier(), &short_statement, proof).unwrap_err();
    assert!(
        matches!(
            error,
            Error::StatementMismatch(StatementMismatch::RecordCount {
                expected: 3,
                actual: 4,
            })
        ),
        "{error:?}"
    );

    if let Some(other_proof) = other_proof {
        let error = verify_aggregate(setup.verifier(), statement, other_proof).unwrap_err();
        assert!(
            matches!(
                error,
                Error::StatementMismatch(StatementMismatch::ParameterSet {
                    expected: 10,
                    actual: 9,
                })
            ),
            "{error:?}"
        );
    }
}

#[test]
fn default_facade_proves_honest_batches_and_rejects_tampering() {
    assert_eq!(WIRE_VERSION, 1);
    assert_eq!(PROTOCOL_STABILITY, Stability::Stable);
    let proof_512 = honest_batch_and_rejections::<Falcon512>(None);
    honest_batch_and_rejections::<Falcon1024>(Some(&proof_512));
}
