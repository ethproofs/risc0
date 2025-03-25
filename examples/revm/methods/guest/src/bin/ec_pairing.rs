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

use risc0_zkvm::guest::env;

fn main() {
    let inp: Vec<u8> = env::read();
    assert_eq!(inp.len(), 192, "EVM ecPairing precompile input must be 192 bytes");

    let result = revm_precompile::bn128::run_pair(&inp, 45000, 34000, 113000).unwrap();
    assert_eq!(result.gas_used, 79000);

    let output: Vec<u8> = result.bytes.into();

    env::commit(&output);
}
