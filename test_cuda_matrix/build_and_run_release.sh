#!/bin/sh

cd ../cuda_matrix_kernel/
cargo build --release  &&
cd ../test_cuda_matrix &&
cargo run --release
