#!/bin/bash

set -eoux

docker build -f docker/cuda-prover.Dockerfile . -t risc0-groth16-cuda-prover
