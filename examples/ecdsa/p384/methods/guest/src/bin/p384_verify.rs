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

use p384::{
    ecdsa::{signature::{hazmat::{PrehashSigner, PrehashVerifier}, Signer, Verifier}, Signature, VerifyingKey},
    elliptic_curve::{point::Double, sec1::{FromEncodedPoint, ToEncodedPoint}, PrimeField},
    EncodedPoint,
};
use risc0_zkvm::guest::env;
use risc0_zkvm::sha::rust_crypto::Digest;

fn main() {
    // TODO: Quick tests of field inversion
    let one = p384::FieldElement::ONE;
    let two = one + one;
    let two_inv = two.invert().unwrap();
    assert_eq!(one, two * two_inv);

    let one = p384::Scalar::ONE;
    let two = one + one;
    let two_inv = two.invert().unwrap();
    assert_eq!(one, two * two_inv);

    // TODO: Tests based on the RustCrypto tests
    assert_eq!(p384::FieldElement::from(2u32) * p384::FieldElement::TWO_INV, p384::FieldElement::ONE);
    assert_eq!(p384::Scalar::from(2u32) * p384::Scalar::TWO_INV, p384::Scalar::ONE);

    assert!(p384::FieldElement::S < 128);
    let two_to_s = 1u128 << p384::FieldElement::S;
    // ROOT_OF_UNITY^{2^s} mod m == 1
    assert_eq!(
        p384::FieldElement::ROOT_OF_UNITY.pow_vartime(&[
            (two_to_s & 0xFFFFFFFFFFFFFFFF) as u64,
            (two_to_s >> 64) as u64,
            0,
            0
        ]),
        p384::FieldElement::ONE
    );
    // MULTIPLICATIVE_GENERATOR^{t} mod m == ROOT_OF_UNITY
    let T: [u64; 6] = [
        0x000000007fffffff,
        0x7fffffff80000000,
        0xffffffffffffffff,
        0xffffffffffffffff,
        0xffffffffffffffff,
        0x7fffffffffffffff,
    ];
    assert_eq!(
        p384::FieldElement::MULTIPLICATIVE_GENERATOR.pow_vartime(&T),
        p384::FieldElement::ROOT_OF_UNITY
    );
    // DELTA^{t} mod m == 1
    assert_eq!(p384::FieldElement::DELTA.pow_vartime(&T), p384::FieldElement::ONE);
    // Same for Scalar
    assert!(p384::Scalar::S < 128);
    let two_to_s = 1u128 << p384::Scalar::S;
    // ROOT_OF_UNITY^{2^s} mod m == 1
    assert_eq!(
        p384::Scalar::ROOT_OF_UNITY.pow_vartime(&[
            (two_to_s & 0xFFFFFFFFFFFFFFFF) as u64,
            (two_to_s >> 64) as u64,
            0,
            0
        ]),
        p384::Scalar::ONE
    );
    // MULTIPLICATIVE_GENERATOR^{t} mod m == ROOT_OF_UNITY
    let T: [u64; 6] = [
        0x76760cb5666294b9,
        0xac0d06d9245853bd,
        0xe3b1a6c0fa1b96ef,
        0xffffffffffffffff,
        0xffffffffffffffff,
        0x7fffffffffffffff,
    ];
    assert_eq!(
        p384::Scalar::MULTIPLICATIVE_GENERATOR.pow_vartime(&T),
        p384::Scalar::ROOT_OF_UNITY
    );
    // DELTA^{t} mod m == 1
    assert_eq!(p384::Scalar::DELTA.pow_vartime(&T), p384::Scalar::ONE);

    assert_eq!(p384::FieldElement::ROOT_OF_UNITY * p384::FieldElement::ROOT_OF_UNITY_INV, p384::FieldElement::ONE);
    assert_eq!(p384::Scalar::ROOT_OF_UNITY * p384::Scalar::ROOT_OF_UNITY_INV, p384::Scalar::ONE);

    // field element square root tests.
    for &n in &[1u64, 4, 9, 16, 25, 36, 49, 64] {
        let fe = p384::FieldElement::from(n);
        let sqrt = fe.sqrt().unwrap();
        assert_eq!(sqrt.square(), fe);
    }
    for &n in &[1u64, 4, 9, 16, 25, 36, 49, 64] {
        let fe = p384::Scalar::from(n);
        let sqrt = fe.sqrt().unwrap();
        assert_eq!(sqrt.square(), fe);
    }

    // field element inversion tests.
    let one = p384::FieldElement::ONE;
    assert_eq!(one.invert().unwrap(), one);
    let three = one + &one + &one;
    let inv_three = three.invert().unwrap();
    assert_eq!(three * &inv_three, one);
    let minus_three = -three;
    let inv_minus_three = minus_three.invert().unwrap();
    assert_eq!(inv_minus_three, -inv_three);
    assert_eq!(three * &inv_minus_three, -one);
    let one = p384::Scalar::ONE;
    assert_eq!(one.invert().unwrap(), one);
    let three = one + &one + &one;
    let inv_three = three.invert().unwrap();
    assert_eq!(three * &inv_three, one);
    let minus_three = -three;
    let inv_minus_three = minus_three.invert().unwrap();
    assert_eq!(inv_minus_three, -inv_three);
    assert_eq!(three * &inv_minus_three, -one);

    // field element identity tests.
    let zero = p384::FieldElement::ZERO;
    let one = p384::FieldElement::ONE;
    assert_eq!(zero.add(&zero), zero);
    assert_eq!(one.add(&zero), one);
    let one = p384::FieldElement::ONE;
    assert_eq!(one.multiply(&one), one);
    let zero = p384::Scalar::ZERO;
    let one = p384::Scalar::ONE;
    assert_eq!(zero.add(&zero), zero);
    assert_eq!(one.add(&zero), one);
    let one = p384::Scalar::ONE;
    assert_eq!(one.multiply(&one), one);

    // From impl_projective_arithmetic_tests
    let basepoint_affine = p384::AffinePoint::GENERATOR;
    let basepoint_projective = p384::ProjectivePoint::GENERATOR;
    assert_eq!(p384::ProjectivePoint::from(basepoint_affine), basepoint_projective,);
    assert_eq!(basepoint_projective.to_affine(), basepoint_affine);
    assert!(!bool::from(basepoint_projective.to_affine().is_identity()));
    assert!(bool::from(p384::ProjectivePoint::IDENTITY.to_affine().is_identity()));

    let identity = p384::ProjectivePoint::IDENTITY;
    let generator = p384::ProjectivePoint::GENERATOR;
    assert_eq!(identity + &generator, generator);
    assert_eq!(generator + &identity, generator);

    let identity = p384::ProjectivePoint::IDENTITY;
    let basepoint_affine = p384::AffinePoint::GENERATOR;
    let basepoint_projective = p384::ProjectivePoint::GENERATOR;

    assert_eq!(identity + &basepoint_affine, basepoint_projective);
    assert_eq!(
        basepoint_projective + &basepoint_affine,
        basepoint_projective + &basepoint_projective
    );

    /// Assert that the provided projective point matches the given test vector.
    // TODO(tarcieri): use coordinate APIs. See zkcrypto/group#30
    macro_rules! assert_point_eq {
        ($actual:expr, $expected:expr) => {
            let (expected_x, expected_y) = $expected;

            let point = $actual.to_affine().to_encoded_point(false);
            let (actual_x, actual_y) = match point.coordinates() {
                p384::elliptic_curve::sec1::Coordinates::Uncompressed { x, y } => (x, y),
                _ => unreachable!(),
            };

            assert_eq!(&expected_x, actual_x.as_slice());
            assert_eq!(&expected_y, actual_y.as_slice());
        };
    }

    let generator = p384::ProjectivePoint::GENERATOR;
    let mut p = generator;
    for i in 0..p384::test_vectors::group::ADD_TEST_VECTORS.len() {
        assert_point_eq!(p, p384::test_vectors::group::ADD_TEST_VECTORS[i]);
        p += &generator;
    }

    let basepoint_affine = p384::AffinePoint::GENERATOR;
    let basepoint_projective = p384::ProjectivePoint::GENERATOR;
    assert_eq!(p384::ProjectivePoint::from(basepoint_affine), basepoint_projective,);
    assert_eq!(basepoint_projective.to_affine(), basepoint_affine);
    assert!(!bool::from(basepoint_projective.to_affine().is_identity()));
    assert!(bool::from(p384::ProjectivePoint::IDENTITY.to_affine().is_identity()));

    let identity = p384::ProjectivePoint::IDENTITY;
    let generator = p384::ProjectivePoint::GENERATOR;
    assert_eq!(identity + &generator, generator);
    assert_eq!(generator + &identity, generator);

    let identity = p384::ProjectivePoint::IDENTITY;
    let basepoint_affine = p384::AffinePoint::GENERATOR;
    let basepoint_projective = p384::ProjectivePoint::GENERATOR;
    assert_eq!(identity + &basepoint_affine, basepoint_projective);
    assert_eq!(
        basepoint_projective + &basepoint_affine,
        basepoint_projective + &basepoint_projective
    );

    let generator = p384::ProjectivePoint::GENERATOR;
    let mut p = generator;
    for i in 0..p384::test_vectors::group::ADD_TEST_VECTORS.len() {
        assert_point_eq!(p, p384::test_vectors::group::ADD_TEST_VECTORS[i]);
        p += &generator;
    }

    let generator = p384::ProjectivePoint::GENERATOR;
    let p0 = generator + p384::ProjectivePoint::IDENTITY;
    let p1 = generator + p384::AffinePoint::IDENTITY;
    assert_eq!(p0, p1);

    let generator = p384::ProjectivePoint::GENERATOR;
    let mut p = generator;
    for i in 0..2 {
        assert_point_eq!(p, p384::test_vectors::group::ADD_TEST_VECTORS[i]);
        p = p.double();
    }

    let generator = p384::ProjectivePoint::GENERATOR;
    assert_eq!(generator + &generator, generator.double());

    let basepoint_affine = p384::AffinePoint::GENERATOR;
    let basepoint_projective = p384::ProjectivePoint::GENERATOR;
    assert_eq!(
        (basepoint_projective + &basepoint_projective) - &basepoint_projective,
        basepoint_projective
    );
    assert_eq!(
        (basepoint_projective + &basepoint_affine) - &basepoint_affine,
        basepoint_projective
    );

    let generator = p384::ProjectivePoint::GENERATOR;
    assert_eq!(generator.double() - &generator, generator);

    let generator = p384::ProjectivePoint::GENERATOR;
    for (k, coords) in p384::test_vectors::group::ADD_TEST_VECTORS
        .iter()
        .enumerate()
        .map(|(k, coords)| (<p384::Scalar>::from(k as u64 + 1), *coords))
        .chain(p384::test_vectors::group::MUL_TEST_VECTORS.iter().cloned().map(|(k, x, y)| {
            (
                <p384::Scalar>::from_repr(
                    primeorder::generic_array::GenericArray::clone_from_slice(&k),
                )
                .unwrap(),
                (x, y),
            )
        }))
    {
        let p = generator * &k;
        assert_point_eq!(p, coords);
    }

    // Tests from p384, first from ecdsa.rs
    // rfc6979
    let x = hex_literal::hex!("6b9d3dad2e1b8c1c05b19875b6659f4de23c3b667bf297ba9aa47740787137d896d5724e4c70a825f872c9ea60d2edf5");
    let signer = p384::ecdsa::SigningKey::from_bytes(&x.into()).unwrap();
    let signature: Signature = signer.sign(b"sample");
    assert_eq!(
        signature.to_bytes().as_slice(),
        &hex_literal::hex!(
            "94edbb92a5ecb8aad4736e56c691916b3f88140666ce9fa73d64c4ea95ad133c81a648152e44acf96e36dd1e80fabe46
            99ef4aeb15f178cea1fe40db2603138f130e740a19624526203b6351d0a3a94fa329c145786e679e7b82c71a38628ac8"
        )
    );
    let signature: Signature = signer.sign(b"test");
    assert_eq!(
        signature.to_bytes().as_slice(),
        &hex_literal::hex!(
            "8203b63d3c853e8d77227fb377bcf7b7b772e97892a80f36ab775d509d7a5feb0542a7f0812998da8f1dd3ca3cf023db
            ddd0760448d42d8a43af45af836fce4de8be06b485e9b61b827c2f13173923e06a739f040649a667bf3b828246baa5a5"
        )
    );

    // prehash_signer_signing_with_sha256
    let x = hex_literal::hex!("6b9d3dad2e1b8c1c05b19875b6659f4de23c3b667bf297ba9aa47740787137d896d5724e4c70a825f872c9ea60d2edf5");
    let signer = p384::ecdsa::SigningKey::from_bytes(&x.into()).unwrap();
    let digest = risc0_zkvm::sha::rust_crypto::Sha256::digest(b"test");
    let signature: Signature = signer.sign_prehash(&digest).unwrap();
    assert_eq!(
        signature.to_bytes().as_slice(),
        &hex_literal::hex!(
            "010c3ab1a300f8c9d63eafa9a41813f0c5416c08814bdfc0236458d6c2603d71c4941f4696e60aff5717476170bb6ab4
            03c4ad6274c61691346b2178def879424726909af308596ffb6355a042f48a114e2eb28eaa6918592b4727961057c0c1"
        )
    );

    // prehash_signer_verification_with_sha256
    let verifier = VerifyingKey::from_affine(
        p384::AffinePoint::from_encoded_point(
            &EncodedPoint::from_affine_coordinates(
                primeorder::generic_array::GenericArray::from_slice(&hex_literal::hex!("0400193b21f07cd059826e9453d3e96dd145041c97d49ff6b7047f86bb0b0439e909274cb9c282bfab88674c0765bc75")),
                primeorder::generic_array::GenericArray::from_slice(&hex_literal::hex!("f70d89c52acbc70468d2c5ae75c76d7f69b76af62dcf95e99eba5dd11adf8f42ec9a425b0c5ec98e2f234a926b82a147")),
                false,
            ),
        ).unwrap()
    ).unwrap();
    let signature = Signature::from_scalars(
        primeorder::generic_array::GenericArray::clone_from_slice(&hex_literal::hex!("b11db00cdaf53286d4483f38cd02785948477ed7ebc2ad609054551da0ab0359978c61851788aa2ec3267946d440e878")),
        primeorder::generic_array::GenericArray::clone_from_slice(&hex_literal::hex!("16007873c5b0604ce68112a8fee973e8e2b6e3319c683a762ff5065a076512d7c98b27e74b7887671048ac027df8cbf2")),
    ).unwrap();
    let result = verifier.verify_prehash(
        &hex_literal::hex!("bbbd0a5f645d3fda10e288d172b299455f9dff00e0fbc2833e18cd017d7f3ed1"),
        &signature,
    );
    assert!(result.is_ok());

    // prehash_signer_verification_with_sha256 -- different encoded point loading
    let verifier = VerifyingKey::from_encoded_point(
        &EncodedPoint::from_affine_coordinates(
            primeorder::generic_array::GenericArray::from_slice(&hex_literal::hex!("0400193b21f07cd059826e9453d3e96dd145041c97d49ff6b7047f86bb0b0439e909274cb9c282bfab88674c0765bc75")),
            primeorder::generic_array::GenericArray::from_slice(&hex_literal::hex!("f70d89c52acbc70468d2c5ae75c76d7f69b76af62dcf95e99eba5dd11adf8f42ec9a425b0c5ec98e2f234a926b82a147")),
            false,
        ),
    ).unwrap();
    let signature = Signature::from_scalars(
        primeorder::generic_array::GenericArray::clone_from_slice(&hex_literal::hex!("b11db00cdaf53286d4483f38cd02785948477ed7ebc2ad609054551da0ab0359978c61851788aa2ec3267946d440e878")),
        primeorder::generic_array::GenericArray::clone_from_slice(&hex_literal::hex!("16007873c5b0604ce68112a8fee973e8e2b6e3319c683a762ff5065a076512d7c98b27e74b7887671048ac027df8cbf2")),
    ).unwrap();
    let result = verifier.verify_prehash(
        &hex_literal::hex!("bbbd0a5f645d3fda10e288d172b299455f9dff00e0fbc2833e18cd017d7f3ed1"),
        &signature,
    );
    assert!(result.is_ok());

    // signing_secret_key_equivalent
    let raw_sk: [u8; 48] = [
        32, 52, 118, 9, 96, 116, 119, 172, 168, 251, 251, 197, 230, 33, 132, 85, 243, 25, 150,
        105, 121, 46, 248, 180, 102, 250, 168, 123, 220, 103, 121, 129, 68, 200, 72, 221, 3,
        102, 30, 237, 90, 198, 36, 97, 52, 12, 234, 150,
    ];
    let seck = p384::SecretKey::from_bytes(&raw_sk.into()).unwrap();
    let sigk = p384::ecdsa::SigningKey::from_bytes(&raw_sk.into()).unwrap();
    assert_eq!(seck.to_bytes().as_slice(), &raw_sk);
    assert_eq!(sigk.to_bytes().as_slice(), &raw_sk);

    // Now from p384's scalar.rs
    // from_to_bytes_roundtrip
    let k: u64 = 42;
    let mut bytes = p384::FieldBytes::default();
    bytes[40..].copy_from_slice(k.to_be_bytes().as_ref());
    let scalar = p384::Scalar::from_repr(bytes).unwrap();
    assert_eq!(bytes, scalar.to_bytes());

    // multiply
    let one = p384::Scalar::ONE;
    let two = one + one;
    let three = two + one;
    let six = three + three;
    assert_eq!(six, two * three);
    let minus_two = -two;
    let minus_three = -three;
    assert_eq!(two, -minus_two);
    assert_eq!(minus_three * minus_two, minus_two * minus_three);
    assert_eq!(six, minus_two * minus_three);

    // Decode the verifying key, message, and signature from the inputs.
    let (encoded_verifying_key, message, signature): (EncodedPoint, Vec<u8>, Signature) =
        env::read();
    let verifying_key = VerifyingKey::from_encoded_point(&encoded_verifying_key).unwrap();

    // Verify the signature, panicking if verification fails.
    verifying_key
        .verify(&message, &signature)
        .expect("ECDSA signature verification failed");

    // Commit to the journal the verifying key and message that was signed.
    env::commit(&(encoded_verifying_key, message));

    // TODO: Deliberately Failing:
    assert_eq!(one, two, "TODO: SUCCESS! (because this _should_ fail)");
}
