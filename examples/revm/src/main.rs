// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// TODO: This file

use revm_methods::{EC_ADD_ELF, EC_ADD_ID, EC_MUL_ELF, EC_MUL_ID};
use risc0_zkvm::{default_prover, ExecutorEnv, Receipt};

/// Prove and get the receipt for an example of an elliptic curve addition
///
/// Corresponds to the EVM ecAdd precompile, see https://www.evm.codes/precompiled?fork=cancun#0x06
fn prove_add() -> Receipt {
    // TODO: Use nontrivial points
    // TODO: More efficient ways to initialize, but who cares, right?
    let x0 = [0u8; 32];
    let y0 = [0u8; 32];
    let x1 = [0u8; 32];
    let y1 = [0u8; 32];
    let mut input: Vec<u8> = x0.into();
    input.extend_from_slice(&y0);
    input.extend_from_slice(&x1);
    input.extend_from_slice(&y1);
    // TODO: Deprecate, right?
    // // g1 and g2 are the generators from EIP-197
    // let g1_compressed =
    //     hex::decode("020000000000000000000000000000000000000000000000000000000000000001")
    //         .expect("valid hex");
    // let g2_compressed = hex::decode("0A04D4BF3239F77CEE7B47C7245E9281B3E9C1182D6381A87BBF81F9F2A6254B731DF569CDA95E060BEE91BA69B3F2D103658A7AEA6B10E5BDC761E5715E7EE4BB").expect("valid hex");
    // // Factors are arbitrary and were chosen at random
    // let a = hex::decode("9c0d02eaaf8e7e7ad09595ef6e3b896f8915124ba5bef9287f0997557580caeb")
    //     .expect("valid hex");
    // let b = hex::decode("db6764642f7bb1f415d93fcd5aace586161ec2e4305f0d6fb57dbabf1d141a5b")
    //     .expect("valid hex");
    // let input = bn254_core::Inputs {
    //     g1_compressed,
    //     g2_compressed,
    //     a,
    //     b,
    // };

    let env = ExecutorEnv::builder()
        .write(&input)
        .unwrap()
        .build()
        .unwrap();

    let prover = default_prover();

    prover.prove(env, EC_ADD_ELF).unwrap().receipt
}

/// Prove and get the receipt for an example of an elliptic curve multiplication
///
/// Corresponds to the EVM ecMul precompile, see https://www.evm.codes/precompiled?fork=cancun#0x07
fn prove_mul() -> Receipt {
    // TODO: Use nontrivial points
    // TODO: More efficient ways to initialize, but who cares, right?
    let x0 = [0u8; 32];
    let y0 = [0u8; 32];
    let s = [47u8; 32];
    let mut input: Vec<u8> = x0.into();
    input.extend_from_slice(&y0);
    input.extend_from_slice(&s);
    // TODO: Deprecate, right?
    // // g1 and g2 are the generators from EIP-197
    // let g1_compressed =
    //     hex::decode("020000000000000000000000000000000000000000000000000000000000000001")
    //         .expect("valid hex");
    // let g2_compressed = hex::decode("0A04D4BF3239F77CEE7B47C7245E9281B3E9C1182D6381A87BBF81F9F2A6254B731DF569CDA95E060BEE91BA69B3F2D103658A7AEA6B10E5BDC761E5715E7EE4BB").expect("valid hex");
    // // Factors are arbitrary and were chosen at random
    // let a = hex::decode("9c0d02eaaf8e7e7ad09595ef6e3b896f8915124ba5bef9287f0997557580caeb")
    //     .expect("valid hex");
    // let b = hex::decode("db6764642f7bb1f415d93fcd5aace586161ec2e4305f0d6fb57dbabf1d141a5b")
    //     .expect("valid hex");
    // let input = bn254_core::Inputs {
    //     g1_compressed,
    //     g2_compressed,
    //     a,
    //     b,
    // };

    let env = ExecutorEnv::builder()
        .write(&input)
        .unwrap()
        .build()
        .unwrap();

    let prover = default_prover();

    prover.prove(env, EC_MUL_ELF).unwrap().receipt
}

fn main() {
    let receipt = prove_add();
    receipt.verify(EC_ADD_ID).unwrap();

    let result: Vec<u8> = receipt
        .journal
        .decode()
        .expect("Journal should contain a Vec with the `bytes` of the PrecompileResult");  // TODO: Not quite accurate now
    println!("Result of EC Add: {result:?}");

    let receipt = prove_mul();
    receipt.verify(EC_MUL_ID).unwrap();

    let result: Vec<u8> = receipt
        .journal
        .decode()
        .expect("Journal should contain a Vec with the `bytes` of the PrecompileResult");  // TODO: Not quite accurate now
    println!("Result of EC Mul: {result:?}");
}

#[test]
fn test_add() {
    let result: Vec<u8> = prove_add().journal.decode().unwrap();
    assert_eq!(
        result,  // TODO: Real test
        vec![
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ],
        "TODO: Message on failure"
    );
}

#[test]
fn test_mul() {
    let result: Vec<u8> = prove_mul().journal.decode().unwrap();
    assert_eq!(
        result,  // TODO: Real test
        vec![
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ],
        "TODO: Message on failure"
    );
}
