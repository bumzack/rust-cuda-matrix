#!/bin/sh

cd cuda_matrix
cargo build --release &&
cd .. &&
cargo run --release

