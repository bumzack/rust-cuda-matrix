#!/bin/sh

cd cuda_matrix
cargo build &&
cd .. &&
cargo run

