use std::error::Error;
use std::ffi::CString;
use std::time::Instant;
use crate::PTX_CODE;

use crate::matrix_utils::print_matrix;
use rustacuda::prelude::*;
use std::fmt;

pub fn invert_matrix_2D2D() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    let nx = 4;
    let nxy = nx * nx;

    let mut matrix_a = vec![0f32; nxy];
    let mut blupp = 1f32;
    for elem in matrix_a.iter_mut() {
        *elem = blupp;
        blupp = blupp + 1.0;
    }
    // ATTENTION:   set one element to different value, otherwise it is a singular matrix
    let idx = nx / 2 * nx + nx / 2;
    matrix_a[idx] = matrix_a[idx] + 23.0;         // just do it!!!

    // make a unit matrix
    let mut matrix_i = vec![0f32; nxy];
    for r in 0..nx {
        matrix_i[r * nx + r] = 1.0;
    }

    println!("orignal matrix_a: ");
    print_matrix(&matrix_a, nx, nx);
    println!("orignal matrix_i: ");
    print_matrix(&matrix_i, nx, nx);

    // Create a context associated to this device
    let _context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let mut d_matrix_a = DeviceBuffer::from_slice(&matrix_a)?;
    let mut d_matrix_i = DeviceBuffer::from_slice(&matrix_i)?;


    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("/tmp/ptx-builder-0.5/cuda_matrix/9132ac0994b05664/nvptx64-nvidia-cuda/release/cuda_matrix.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;


    let blocksize = 1;
    let threads_per_block = (blocksize, blocksize, 1);

    let b = (blocksize, blocksize, 1);
    let block = (b.0 as u32, b.1 as u32, b.2 as u32);

    let g = (
        (nx as i32 + blocksize as i32 - 1) / blocksize as i32,
        (nx as i32 + blocksize as i32 - 1) / blocksize as i32,
        1 as i32,
    );
    let grid = (g.0 as u32, g.1 as u32, 1 as u32);

    println!("block = {:?}, grid = {:?}", block, grid);

    let start = Instant::now();

    for i in 0..nx {
        unsafe {
            // Launch the `add` function with one block containing four threads on the stream.
            let res = launch!(module.nodiag_normalize<<<grid, block, 0, stream>>>(
                d_matrix_a.as_device_ptr(),
                d_matrix_i.as_device_ptr(),
                nx,
                i,
                block.0,
                block.1
            ));

            match res {
                Ok(_o) => (),
                Err(e) => println!("an error occured: {}", e),
            }
        }
        unsafe {
            // Launch the `add` function with one block containing four threads on the stream.
            let res = launch!(module.diag_normalize<<<grid, block, 0, stream>>>(
                d_matrix_a.as_device_ptr(),
                d_matrix_i.as_device_ptr(),
                nx,
                i,
                block.0,
                block.1
            ));

            match res {
                Ok(_o) => (),
                Err(e) => println!("an error occured: {}", e),
            }
        }
        unsafe {
            // Launch the `add` function with one block containing four threads on the stream.
            let res = launch!(module.gaussjordan<<<grid, block, 0, stream>>>(
                d_matrix_a.as_device_ptr(),
                d_matrix_i.as_device_ptr(),
                nx,
                i,
                block.0,
                block.1
            ));

            match res {
                Ok(_o) => (),
                Err(e) => println!("an error occured: {}", e),
            }
        }
        unsafe {
            // Launch the `add` function with one block containing four threads on the stream.
            let res = launch!(module.set_zero<<<grid, block, 0, stream>>>(
                d_matrix_a.as_device_ptr(),
                d_matrix_i.as_device_ptr(),
                nx,
                i,
                block.0,
                block.1
            ));

            match res {
                Ok(_o) => (),
                Err(e) => println!("an error occured: {}", e),
            }
        }
    }
    stream.synchronize()?;

    let duration_cuda = start.elapsed();

    d_matrix_a.copy_to(&mut matrix_a)?;
    d_matrix_i.copy_to(&mut matrix_i)?;
    println!("duration gpu  invert_matrix_2D2D: {:?}", duration_cuda);

    println!("gpu result  inverted matrix: \n\n");
    print_matrix(&matrix_i, nx, nx);

    println!("former input matrix  ");
    print_matrix(&matrix_a, nx, nx);

//
//    let start_cpu = Instant::now();
//    let res_cpu = invert_matrix_cpu(&matrix_a, nx, ny);
//    let duration_cpu = start_cpu.elapsed();
//
//    println!("duration cpu: {:?}", duration_cpu);
//
//    for x in 0..res_cpu.len() {
//        // assert_eq!(res_cpu[x], out_host[x]);
//    }
    // The kernel launch is asynchronous, so we wait for the kernel to finish executing

    Ok(())
}

impl fmt::Display for Matrix {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\nrows: {}, cols: {}\n", self.rows, self.cols)?;
        for row in 0..self.rows {
            for col in 0..self.cols {
                write!(f, " {} ", self.get(row, col))?;
            }
            write!(f, "\n ")?;
        }
        write!(f, "\n ")
    }
}

pub fn test_matrix_invert_cpu() {
    let mut m = Matrix::zero(3, 3);
    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(0, 2, 3.0);
    m.set(1, 0, 0.0);
    m.set(1, 1, 1.0);
    m.set(1, 2, 4.0);
    m.set(2, 0, 5.0);
    m.set(2, 1, 6.0);
    m.set(2, 2, 0.0);

    let mut expected = Matrix::zero(3, 3);
    expected.set(0, 0, -24.0);
    expected.set(0, 1, 18.0);
    expected.set(0, 2, 5.0);
    expected.set(1, 0, 20.0);
    expected.set(1, 1, -15.0);
    expected.set(1, 2, -4.0);
    expected.set(2, 0, -5.0);
    expected.set(2, 1, 4.0);
    expected.set(2, 2, 1.0);

    // calculate the inverse and compare with expected result
    let inv = matrix_invert_cpu(&m).unwrap();
    assert_eq!(expected, inv);
    println!("orignal: {}", m);
    println!("inverted: {}", inv);
}

#[derive(Debug, PartialEq, Clone)]
pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn one(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows: rows,
            cols: cols,
            data: vec![1.0; cols * rows],
        }
    }

    pub fn zero(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows: rows,
            cols: cols,
            data: vec![0.0; cols * rows],
        }
    }

    pub fn identiy(rows: usize) -> Matrix {
        let mut m = Matrix::zero(rows, rows);
        for i in 0..rows {
            m.set(i, i, 1.0);
        }
        m
    }

    pub fn get_rows(&self) -> usize {
        self.rows
    }

    pub fn get_cols(&self) -> usize {
        self.cols
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) -> &mut Matrix {
        self.data[row * self.cols + col] = value;
        self
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }
}

pub fn matrix_invert_cpu(mat_a: &Matrix) -> Result<Matrix, MathError> {
    if mat_a.rows != mat_a.cols {
        return Err(MathError::MatrixNotInvertableNotSquare);
    }
    let rows = mat_a.rows;
    let mut cols = mat_a.cols;

    // helper matrix for inverting
    let mut dummy = Matrix::zero(rows, 2 * cols);

    // copy matrix a to dummy (left half of dummy)
    for row in 0..rows {
        for col in 0..cols {
            dummy.set(row, col, mat_a.get(row, col));
        }
    }
    // set identiy matrix elements
    for row in 0..rows {
        dummy.set(row, cols + row, 1.0);
    }
    // apply all transformations to the identiy matrix as well
    cols = 2 * mat_a.cols;

    let mut tmp: f32 = 0.0;
    for row in 0..rows {
        // transform to an upper triangle matrix
        // element in main diagonal is used to divide the current row
        let mut elem = dummy.get(row, row);
        if elem != 0.0 {
            // divide each element in current row by elem -> so A(row,row) = 1
            for col in 0..cols {
                tmp = dummy.get(row, col);
                dummy.set(row, col, tmp / elem);
            }
            // subtract the line row from all the rows below this row
            for row2 in row + 1..rows {
                elem = dummy.get(row2, row);
                for col in 0..cols {
                    // dummy.data[row2][col] = dummy.data[row2][col] - elem * dummy.data[row][col];
                    tmp = dummy.get(row2, col) - elem * dummy.get(row, col);
                    dummy.set(row2, col, tmp);
                }
            }
        } else {
            return Err(MathError::MatrixNotInvertable(row, row, elem as f64));
        }
    }

    // all elements below the main diagonal are 0
    // iterate from the last row to the first row and
    // set the elements right from the diagonal to 0
    // transform to an upper triangle matri
    // element in main diagonal is used to divide the current row

    for row in (1..rows).rev() {
        // transform to an lower triangle matrix
        // subtract the line row from all the rows above  this row

        for row2 in (0..row).rev() {
            let elem = dummy.get(row2, row);
            for col in 0..cols {
                // dummy.data[row2][col] = dummy.data[row2][col] - elem * dummy.data[row][col];
                tmp = dummy.get(row2, col) - elem * dummy.get(row, col);
                dummy.set(row2, col, tmp);
            }
        }
    }

    let mut res = Matrix::zero(rows, rows);
    for row in 0..rows {
        for col in 0..mat_a.cols {
            // res.data[row][col] = dummy.data[row][col + mat_a.cols];
            tmp = dummy.get(row, col + mat_a.get_cols());
            res.set(row, col, tmp);
        }
    }
    Ok(res)
}

#[derive(Debug)]
pub enum MathError {
    MatrixDimensionDontMatch,
    MatrixNotInvertableNotSquare,
    MatrixMulNotSquare,
    MatrixNotInvertable(usize, usize, f64),
}
