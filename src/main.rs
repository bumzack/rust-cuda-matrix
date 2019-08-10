#[macro_use]
extern crate rustacuda;

use crate::matrix_add::{add_matrix_2D1D, add_matrix_2D2D};
use crate::matrix_mul::{mul_matrix_2D1D, mul_matrix_2D2D};
use crate::matrix_utils::test_matrix;
use std::error::Error;

pub mod cuda_utils;
pub mod matrix_add;
pub mod matrix_invert;
pub mod matrix_mul;
pub mod matrix_transpose;
pub mod matrix_utils;

fn main() -> Result<(), Box<dyn Error>> {
    test_matrix()?;
    add_matrix_2D2D()?;
    add_matrix_2D1D()?;
    mul_matrix_2D2D()?;
    mul_matrix_2D1D()?;
    //transpose_matrix_row_2D2D()?;
    //transpose_matrix_col_2D2D()?;
    // transpose_matrix_unroll4C()?;
    Ok(())
}
