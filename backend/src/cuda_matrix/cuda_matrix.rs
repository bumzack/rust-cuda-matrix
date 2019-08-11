use crate::cpu_matrix::cpu_matrix::CpuMatrix;
use rustacuda::prelude::*;
use std::fmt;
use crate::matrix::matrix::Matrix;
#[derive(Debug)]
pub struct CudaMatrix {
    cpu_matrix: CpuMatrix,
    host_matrix: DeviceBuffer<f32>,
}

// TODO: what do we want to display here ...
impl fmt::Display for CudaMatrix {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "\nthe CPU matrix of the CUDA matrix contains: \nrows: {}, cols: {}\n",
            self.cpu_matrix.get_rows(), self.cpu_matrix.get_cols()
        )?;
        for row in 0..self.cpu_matrix.get_rows() {
            for col in 0..self.cpu_matrix.get_cols() {
                write!(f, " {} ", self.cpu_matrix.get(row, col))?;
            }
            write!(f, "\n ")?;
        }
        write!(f, "\n ")
    }
}
