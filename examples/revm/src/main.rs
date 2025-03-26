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

use revm_methods::{EC_ADD_ELF, EC_ADD_ID, EC_MUL_ELF, EC_MUL_ID, EC_PAIRING_ELF, EC_PAIRING_ID};
use risc0_zkvm::{default_prover, ExecutorEnv, Receipt};

/// Prove and get the receipt for an example of an elliptic curve addition
///
/// Corresponds to the EVM ecAdd precompile, see https://www.evm.codes/precompiled?fork=cancun#0x06
fn prove_add(input: Vec<u8>) -> Receipt {
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
fn prove_mul(input: Vec<u8>) -> Receipt {
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

/// Prove and get the receipt for an example of an elliptic curve pairing
///
/// Corresponds to the EVM ecPairing precompile, see https://www.evm.codes/precompiled?fork=cancun#0x08
fn prove_pairing(input: Vec<u8>) -> Receipt {
    let env = ExecutorEnv::builder()
        .write(&input)
        .unwrap()
        .build()
        .unwrap();

    let prover = default_prover();

    prover.prove(env, EC_PAIRING_ELF).unwrap().receipt
}

fn main() {
    // TODO: Use nontrivial points
    // TODO: More efficient ways to initialize, but who cares, right?
    let x0 = [0u8; 32];
    let y0 = [0u8; 32];
    let x1 = [0u8; 32];
    let y1 = [0u8; 32];
    let mut add_input: Vec<u8> = x0.into();
    add_input.extend_from_slice(&y0);
    add_input.extend_from_slice(&x1);
    add_input.extend_from_slice(&y1);
    let receipt = prove_add(add_input);
    receipt.verify(EC_ADD_ID).unwrap();

    let result: Vec<u8> = receipt
        .journal
        .decode()
        .expect("Journal should contain a Vec with the `bytes` of the PrecompileResult");  // TODO: Not quite accurate now
    println!("Result of EC Add: {result:?}");

    // TODO: Use nontrivial points
    // TODO: More efficient ways to initialize, but who cares, right?
    let x0 = [0u8; 32];
    let y0 = [0u8; 32];
    let s = [47u8; 32];
    let mut mul_input: Vec<u8> = x0.into();
    mul_input.extend_from_slice(&y0);
    mul_input.extend_from_slice(&s);
    let receipt = prove_mul(mul_input);
    receipt.verify(EC_MUL_ID).unwrap();

    let result: Vec<u8> = receipt
        .journal
        .decode()
        .expect("Journal should contain a Vec with the `bytes` of the PrecompileResult");  // TODO: Not quite accurate now
    println!("Result of EC Mul: {result:?}");

    // TODO: Use nontrivial points
    // TODO: More efficient ways to initialize, but who cares, right?
    let x0 = hex::decode("2cf44499d5d27bb186308b7af7af02ac5bc9eeb6a3d147c186b21fb1b76e18da").unwrap();
    let mut pair_input: Vec<u8> = x0.into();
    pair_input.extend_from_slice(&hex::decode("2c0f001f52110ccfe69108924926e45f0b0c868df0e7bde1fe16d3242dc715f6").unwrap());
    pair_input.extend_from_slice(&hex::decode("1fb19bb476f6b9e44e2a32234da8212f61cd63919354bc06aef31e3cfaff3ebc").unwrap());
    pair_input.extend_from_slice(&hex::decode("22606845ff186793914e03e21df544c34ffe2f2f3504de8a79d9159eca2d98d9").unwrap());
    pair_input.extend_from_slice(&hex::decode("2bd368e28381e8eccb5fa81fc26cf3f048eea9abfdd85d7ed3ab3698d63e4f90").unwrap());
    pair_input.extend_from_slice(&hex::decode("2fe02e47887507adf0ff1743cbac6ba291e66f59be6bd763950bb16041a0a85e").unwrap());
    pair_input.extend_from_slice(&hex::decode("0000000000000000000000000000000000000000000000000000000000000001").unwrap());
    pair_input.extend_from_slice(&hex::decode("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd45").unwrap());
    pair_input.extend_from_slice(&hex::decode("1971ff0471b09fa93caaf13cbf443c1aede09cc4328f5a62aad45f40ec133eb4").unwrap());
    pair_input.extend_from_slice(&hex::decode("091058a3141822985733cbdddfed0fd8d6c104e9e9eff40bf5abfef9ab163bc7").unwrap());
    pair_input.extend_from_slice(&hex::decode("2a23af9a5ce2ba2796c1f4e453a370eb0af8c212d9dc9acd8fc02c2e907baea2").unwrap());
    pair_input.extend_from_slice(&hex::decode("23a8eb0b0996252cb548a4487da97b02422ebc0e834613f954de6c7e0afdc1fc").unwrap());
    let receipt = prove_pairing(pair_input);
    receipt.verify(EC_PAIRING_ID).unwrap();

    let result: Vec<u8> = receipt
        .journal
        .decode()
        .expect("Journal should contain a Vec with the `bytes` of the PrecompileResult");  // TODO: Not quite accurate now
    println!("Result of EC Pairing: {result:?}");
}

#[test]
fn test_add_trivial() {
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
    let result: Vec<u8> = prove_add(input).journal.decode().unwrap();
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
fn test_mul_trivial() {
    // TODO: More efficient ways to initialize, but who cares, right?
    let x0 = [0u8; 32];
    let y0 = [0u8; 32];
    let s = [47u8; 32];
    let mut input: Vec<u8> = x0.into();
    input.extend_from_slice(&y0);
    input.extend_from_slice(&s);
    let result: Vec<u8> = prove_mul(input).journal.decode().unwrap();
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
fn test_pairing_trivial() {
    // TODO: More efficient ways to initialize, but who cares, right?
    let x0 = [0u8; 32];
    let y0 = [0u8; 32];
    let x1 = [0u8; 32];
    let y1 = [0u8; 32];
    let x2 = [0u8; 32];
    let y2 = [0u8; 32];
    let mut input: Vec<u8> = x0.into();
    input.extend_from_slice(&y0);
    input.extend_from_slice(&x1);
    input.extend_from_slice(&y1);
    input.extend_from_slice(&x2);
    input.extend_from_slice(&y2);
    let result: Vec<u8> = prove_pairing(input).journal.decode().unwrap();
    assert_eq!(
        result,  // TODO: Real test
        vec![
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1,
        ],
        "TODO: Message on failure"
    );
}
