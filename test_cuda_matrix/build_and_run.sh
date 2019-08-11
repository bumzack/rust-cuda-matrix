#!/bin/sh

cd ../cuda_matrix_kernel/
cargo build &&
cd ../test_cuda_matrix &&
cargo run
