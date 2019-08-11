extern crate cuda_matrix;

 use std::error::Error;

use cuda_matrix::matrix_ops;

fn main() -> Result<(), Box<dyn Error>> {
    //    test_matrix()?;
    //    add_matrix_2D2D()?;
    //    add_matrix_2D1D()?;
    //    mul_matrix_2D2D()?;
    //    mul_matrix_2D1D()?;
    //transpose_matrix_row_2D2D()?;
    //transpose_matrix_col_2D2D()?;
    // transpose_matrix_unroll4C()?;
    // invert_matrix_2D2D()?;
//    test_matrix_invert_cpu();

    matrix_ops::matrix_invert::invert_matrix_2D2D()?;

    Ok(())

}
