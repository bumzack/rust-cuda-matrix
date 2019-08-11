//#![macro_use]
//extern  crate rustacuda;

extern crate cuda_matrix;

use std::error::Error;

use backend::backend::cpu_backend::CpuBackend;
use backend::backend::cuda_backend::CudaBackend;
use backend::backend::backend::Backend;
use backend::matrix::matrix::Matrix;


fn main_cpu() -> Result<(), Box<dyn Error>> {
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

    // matrix_ops::matrix_invert::invert_matrix_2D2D()?;

    let cpu_backend = CpuBackend::new();
    let mut a = cpu_backend.new_matrix(2, 2);
    a.set(0, 0, 1.0);
    a.set(0, 1, 2.0);
    a.set(1, 0, 3.0);
    a.set(1, 1, 4.0);

    let mut b = cpu_backend.new_matrix(2, 2);
    b.set(0, 0, 11.0);
    b.set(0, 1, 12.0);
    b.set(1, 0, 13.0);
    b.set(1, 1, 14.0);

    let c = cpu_backend.add(&mut a, &mut b);
    println!("a =  {}", a);
    println!("b =  {}", b);
    println!("a +b =  {}", c);

    Ok(())
}

fn main_cuda() -> Result<(), Box<dyn Error>> {

    let cuda_backend = CudaBackend::new();
    let mut a = cuda_backend.new_matrix(2, 2);
    a.set(0, 0, 1.0);
    a.set(0, 1, 2.0);
    a.set(1, 0, 3.0);
    a.set(1, 1, 4.0);

    let mut b = cuda_backend.new_matrix(2, 2);
    b.set(0, 0, 11.0);
    b.set(0, 1, 12.0);
    b.set(1, 0, 13.0);
    b.set(1, 1, 14.0);

    let c = cuda_backend.add(&mut a, &mut b);
    println!("a =  {}", a);
    println!("b =  {}", b);
    println!("a +b =  {}", c);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    main_cpu()?;
    main_cuda()?;

    Ok(())
}
