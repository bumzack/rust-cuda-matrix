use std::error::Error;
use std::ffi::CString;
use std::time::Instant;

use rustacuda::prelude::*;

pub fn invert_matrix_2D1D() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    let nx = 8000;
    let ny = 8000;
    let nxy = nx * ny;

    let mut matrix_a = vec![0f32; nxy];
    let mut blupp = 1f32;
    for elem in matrix_a.iter_mut() {
        *elem = blupp;
        blupp = blupp + 1.0;
    }

    let mut matrix_b = vec![0f32; nxy];
    let mut blupp = 101f32;
    for elem in matrix_b.iter_mut() {
        *elem = blupp;
        blupp = blupp + 1.0;
    }
    let mut matrix_c = vec![0f32; nxy];

    // Create a context associated to this device
    let _context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let mut d_matrix_a = DeviceBuffer::from_slice(&matrix_a)?;
    let mut d_matrix_b = DeviceBuffer::from_slice(&matrix_b)?;
    let mut d_matrix_c = DeviceBuffer::from_slice(&matrix_c)?;

    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("/tmp/ptx-builder-0.5/cuda_matrix/9132ac0994b05664/nvptx64-nvidia-cuda/release/cuda_matrix.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let b = (256, 1, 1);
    let block = (b.0 as u32, b.1 as u32, b.2 as u32);

    let g = (
        (nx as i32 + block.0 as i32 - 1) / block.0 as i32,
        (ny as i32 + block.1 as i32 - 1) / block.1 as i32,
        1 as i32,
    );
    let grid = (g.0 as u32, g.1 as u32, 1 as u32);

    println!("block = {:?}, grid = {:?}", block, grid);

    let start = Instant::now();

    unsafe {
        // Launch the `add` function with one block containing four threads on the stream.
        let res = launch!(module.sumMatrixOnGpu2D1D<<<grid, block, 0, stream>>>(
            d_matrix_a.as_device_ptr(),
            d_matrix_b.as_device_ptr(),
            d_matrix_c.as_device_ptr(),
            nx,
            ny,
            block.0,
            block.1
        ));

        match res {
            Ok(_o) => println!("everything ok"),
            Err(e) => println!("an error occured: {}", e),
        }
    }
    stream.synchronize()?;

    let duration_cuda = start.elapsed();

    let mut out_host = vec![0.0f32; nxy];
    d_matrix_c.copy_to(&mut out_host)?;
    println!("duration gpu  sumMatrixOnGpu2D1D: {:?}", duration_cuda);

    //    println!("out_host  = {:?}", out_host);

    //    let start_cpu = Instant::now();
    //    let res_cpu = add_matrix_cpu(&matrix_a, &matrix_b, nx, ny);
    //    let duration_cpu = start_cpu.elapsed();
    //
    //    println!("duration cpu: {:?}", duration_cpu);
    //
    //    for x in 0..res_cpu.len() {
    //        assert_eq!(res_cpu[x], out_host[x]);
    //    }
    //    // The kernel launch is asynchronous, so we wait for the kernel to finish executing

    // show_device_props(&device);

    Ok(())
}
//
//pub fn invert_matrix_cpu(a: &Vec<f32>, mat_a_col: usize, mat_a_row: usize) -> Vec<f32> {
//    let mut res: Vec<f32> = vec![0f32; mat_a_row * mat_b_col];
//
//    if mat_a_col != mat_a_row {
//        return res;
//    }
//    let rows = mat_a.rows;
//    let mut cols = mat_a.cols;
//
//    // helper matrix for inverting
//    let mut dummy = Matrix::zero(rows, 2 * cols);
//
//    // copy matrix a to dummy (left half of dummy)
//    for row in 0..rows {
//        for col in 0..cols {
//            dummy.set(row, col, mat_a.get(row, col));
//        }
//    }
//    // set identiy matrix elements
//    for row in 0..rows {
//        dummy.set(row, cols + row, 1.0);
//    }
//    // apply all transformations to the identiy matrix as well
//    cols = 2 * mat_a.cols;
//
//    let mut tmp: f32 = 0.0;
//    for row in 0..rows {
//        // transform to an upper triangle matrix
//        // element in main diagonal is used to divide the current row
//        let mut elem = dummy.get(row, row);
//        if elem != 0.0 {
//            // divide each element in current row by elem -> so A(row,row) = 1
//            for col in 0..cols {
//                tmp = dummy.get(row, col);
//                dummy.set(row, col, tmp / elem);
//            }
//            // subtract the line row from all the rows below this row
//            for row2 in row + 1..rows {
//                elem = dummy.get(row2, row);
//                for col in 0..cols {
//                    // dummy.data[row2][col] = dummy.data[row2][col] - elem * dummy.data[row][col];
//                    tmp = dummy.get(row2, col) - elem * dummy.get(row, col);
//                    dummy.set(row2, col, tmp);
//                }
//            }
//        } else {
//            return Err(admin::MathError::MatrixNotInvertable(row, row, elem as f64));
//        }
//    }
//
//    // all elements below the main diagonal are 0
//    // iterate from the last row to the first row and
//    // set the elements right from the diagonal to 0
//    // transform to an upper triangle matri
//    // element in main diagonal is used to divide the current row
//
//    for row in (1..rows).rev() {
//        // transform to an lower triangle matrix
//        // subtract the line row from all the rows above  this row
//
//        for row2 in (0..row).rev() {
//            let elem = dummy.get(row2, row);
//            for col in 0..cols {
//                // dummy.data[row2][col] = dummy.data[row2][col] - elem * dummy.data[row][col];
//                tmp = dummy.get(row2, col) - elem * dummy.get(row, col);
//                dummy.set(row2, col, tmp);
//            }
//        }
//    }
//
//    let mut res = Matrix::zero(rows, rows);
//    for row in 0..rows {
//        for col in 0..mat_a.cols {
//            // res.data[row][col] = dummy.data[row][col + mat_a.cols];
//            tmp = dummy.get(row, col + mat_a.get_cols());
//            res.set(row, col, tmp);
//        }
//    }
//    res
//}
