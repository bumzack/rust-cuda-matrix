use std::error::Error;
use std::ffi::CString;
use std::time::Instant;
use crate::PTX_CODE;

use crate::matrix_utils::print_matrix;
use rustacuda::prelude::*;

fn transpose_matrix_row_2D2D() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    // MAtrix A
    let mat_a_row = 80;
    let mat_a_col = 50;
    let mut matrix_a = vec![0f32; mat_a_row * mat_a_col];
    let mut blupp = 1f32;
    for elem in matrix_a.iter_mut() {
        *elem = blupp;
        blupp = blupp + 1.0;
    }

    // MAtrix B
    let mat_b_row = 50;
    let mat_b_col = 80;
    let mut matrix_b = vec![0f32; mat_b_col * mat_b_row];

    // Create a context associated to this device
    let _context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let mut d_matrix_a = DeviceBuffer::from_slice(&matrix_a)?;
    let mut d_matrix_b = DeviceBuffer::from_slice(&matrix_b)?;

    // Load the module containing the function we want to call
    let module_data = CString::new(PTX_CODE)?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let b = (256, 256, 1);
    let block = (b.0 as u32, b.1 as u32, b.2 as u32);

    let g = (
        (mat_a_col as i32 + block.0 as i32 - 1) / block.0 as i32,
        (mat_a_row as i32 + block.1 as i32 - 1) / block.1 as i32,
        1 as i32,
    );
    let grid = (g.0 as u32, g.1 as u32, 1 as u32);

    println!("block = {:?}, grid = {:?}", block, grid);

    let start = Instant::now();
    let mut success_gpu = true;

    unsafe {
        // Launch the `add` function with one block containing four threads on the stream.
        let res = launch!(module.transpose_matrix_col_2D2D<<<grid, block, 0, stream>>>(
            d_matrix_a.as_device_ptr(),
            d_matrix_b.as_device_ptr(),
            mat_a_row,
            mat_a_col,
            mat_b_row,
            mat_b_col,
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

    let mut out_host = vec![0.0f32; mat_a_row * mat_a_col];
    d_matrix_b.copy_to(&mut out_host)?;

    if success_gpu {
        println!(
            "duration gpu  transpose_matrix_row_2D2D: {:?}",
            duration_cuda
        );
    } else {
        println!("error calculating matrix transpose for gpu ");
    }

    // println!("matrix a  = {:?}",matrix_a);
    // ntln!("matrix b = {:?}",matrix_b);

    //    println!("result gpu ");
    //    print_matrix(&out_host, mat_b_row, mat_b_col);

    let start_cpu = Instant::now();

    let res_cpu = transpose_matrix_cpu(&matrix_a, mat_a_col, mat_a_row);
    let duration_cpu = start_cpu.elapsed();

    //    println!("result cpu  ");
    //    print_matrix(&res_cpu, mat_b_row, mat_b_col);

    println!("duration cpu: {:?}", duration_cpu);

    println!("check if results are equal");
    for x in 0..res_cpu.len() {
        assert_eq!(res_cpu[x], out_host[x]);
    }
    println!("check if results are  equal   ....  yes!!!!");
    // The kernel launch is asynchronous, so we wait for the kernel to finish executing

    // show_device_props(&device);

    Ok(())
}

fn transpose_matrix_col_2D2D() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    // MAtrix A
    let mat_a_row = 80;
    let mat_a_col = 50;
    let mut matrix_a = vec![0f32; mat_a_row * mat_a_col];
    let mut blupp = 1f32;
    for elem in matrix_a.iter_mut() {
        *elem = blupp;
        blupp = blupp + 1.0;
    }

    // MAtrix B
    let mat_b_row = 50;
    let mat_b_col = 80;
    let mut matrix_b = vec![0f32; mat_b_col * mat_b_row];

    // Create a context associated to this device
    let _context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let mut d_matrix_a = DeviceBuffer::from_slice(&matrix_a)?;
    let mut d_matrix_b = DeviceBuffer::from_slice(&matrix_b)?;

    // Load the module containing the function we want to call
    let module_data = CString::new(PTX_CODE)?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let b = (1024, 1024, 1);
    let block = (b.0 as u32, b.1 as u32, b.2 as u32);

    let g = (
        (mat_a_col as i32 + block.0 as i32 - 1) / block.0 as i32,
        (mat_a_row as i32 + block.1 as i32 - 1) / block.1 as i32,
        1 as i32,
    );
    let grid = (g.0 as u32, g.1 as u32, 1 as u32);

    println!("block = {:?}, grid = {:?}", block, grid);

    let start = Instant::now();
    let mut success_gpu = true;

    unsafe {
        // Launch the `add` function with one block containing four threads on the stream.
        let res = launch!(module.transpose_matrix_col_2D2D<<<grid, block, 0, stream>>>(
            d_matrix_a.as_device_ptr(),
            d_matrix_b.as_device_ptr(),
            mat_a_row,
            mat_a_col,
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

    let mut out_host = vec![0.0f32; mat_a_row * mat_a_col];
    d_matrix_b.copy_to(&mut out_host)?;

    if success_gpu {
        println!(
            "duration gpu  transpose_matrix_col_2D2D: {:?}",
            duration_cuda
        );
    } else {
        println!("error calculating matrix transpose for gpu ");
    }

    // println!("matrix a  = {:?}",matrix_a);
    // ntln!("matrix b = {:?}",matrix_b);

    println!("result gpu ");
    print_matrix(&out_host, mat_b_row, mat_b_col);

    let start_cpu = Instant::now();
    let res_cpu = transpose_matrix_cpu(&matrix_a, mat_a_col, mat_a_row);
    let duration_cpu = start_cpu.elapsed();

    println!("result cpu  ");
    print_matrix(&res_cpu, mat_b_row, mat_b_col);

    println!("duration cpu: {:?}", duration_cpu);

    println!("check if results are equal");
    for x in 0..res_cpu.len() {
        assert_eq!(res_cpu[x], out_host[x]);
    }
    println!("check if results are  equal   ....  yes!!!!");
    // The kernel launch is asynchronous, so we wait for the kernel to finish executing

    // show_device_props(&device);

    Ok(())
}

//
//fn transpose_matrix_unroll4C() -> Result<(), Box<dyn Error>> {
//    // Initialize the CUDA API
//    rustacuda::init(CudaFlags::empty())?;
//
//    // Get the first device
//    let device = Device::get_device(0)?;
//
//    // MAtrix A
//    let mat_a_row = 128;
//    let mat_a_col = 64;
//    let mut matrix_a = vec![0f32; mat_a_row * mat_a_col];
//    let mut blupp = 1f32;
//    for elem in matrix_a.iter_mut() {
//        *elem = blupp;
//        blupp = blupp + 1.0;
//    }
//
//    // MAtrix B
//    let mat_b_row = 64;
//    let mat_b_col = 128;
//    let mut matrix_b = vec![0f32; mat_b_col * mat_b_row];
//
//    // Create a context associated to this device
//    let _context =
//        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
//
//    let mut d_matrix_a = DeviceBuffer::from_slice(&matrix_a)?;
//    let mut d_matrix_b = DeviceBuffer::from_slice(&matrix_b)?;
//
//    // Load the module containing the function we want to call
//    let module_data = CString::new(PTX_CODE)?;
//    let module = Module::load_from_string(&module_data)?;
//
//    // Create a stream to submit work to
//    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
//
//    let b = (16, 16, 1);
//    let block = (b.0 as u32, b.1 as u32, b.2 as u32);
//
//    let mut g = (
//        (mat_a_col as i32 + block.0 as i32 - 1) / block.0 as i32,
//        (mat_a_row as i32 + block.1 as i32 - 1) / block.1 as i32,
//        1 as i32,
//    );
//    g.0 = (mat_a_col as i32 + block.0 as i32 * 4 - 1) / (block.0 as i32 * 4);
//
//    let grid = (g.0 as u32, g.1 as u32, 1 as u32);
//
//    println!("block = {:?}, grid = {:?}", block, grid);
//
//    let start = Instant::now();
//    let mut success_gpu = true;
//
//    unsafe {
//        // Launch the `add` function with one block containing four threads on the stream.
//        let res = launch!(module.transpose_matrix_unroll4C<<<grid, block, 0, stream>>>(
//            d_matrix_a.as_device_ptr(),
//            d_matrix_b.as_device_ptr(),
//            mat_a_col,
//            mat_a_row,
//            block.0,
//            block.1
//        ));
//
//        match res {
//            Ok(_o) => println!("everything ok"),
//            Err(e) => println!("an error occured: {}", e),
//        }
//    }
//    stream.synchronize()?;
//
//    let duration_cuda = start.elapsed();
//
//    let mut out_host = vec![0.0f32; mat_a_row * mat_a_col];
//    d_matrix_b.copy_to(&mut out_host)?;
//
//    if success_gpu {
//        println!("duration gpu  transpose_matrix_2D1D: {:?}", duration_cuda);
//    } else {
//        println!("error calculating matrix transpose for gpu ");
//    }
//
//    println!("matrix a  = {:?}", matrix_a);
//    println!("matrix b = {:?}", matrix_b);
//
//    println!("result gpu ");
//    print_matrix(&out_host, mat_b_row, mat_b_col);
//
//    let start_cpu = Instant::now();
//    let mut success_cpu = false;
//    let res_cpu = transpose_matrix_cpu(
//        &matrix_a,
//        mat_a_row,
//        mat_a_col,
//        mat_b_row,
//        mat_b_col,
//        &mut success_cpu,
//    );
//    let duration_cpu = start_cpu.elapsed();
//
//    println!("result cpu  ");
//    print_matrix(&res_cpu, mat_b_row, mat_b_col);
//
//    if success_cpu {
//        println!("duration cpu: {:?}", duration_cpu);
//    } else {
//        println!("error calculating matrix transpose for cpu ");
//    }
//
//    println!("check if results are equal");
//    for x in 0..res_cpu.len() {
//        assert_eq!(res_cpu[x], out_host[x]);
//    }
//    println!("check if results are  equal   ....  yes!!!!");
//    // The kernel launch is asynchronous, so we wait for the kernel to finish executing
//
//    // show_device_props(&device);
//
//    Ok(())
//}

fn transpose_matrix_cpu(a: &Vec<f32>, mat_a_col: usize, mat_a_row: usize) -> Vec<f32> {
    let mut res: Vec<f32> = vec![0f32; mat_a_col * mat_a_row];

    println!("res.len() = {}", res.len());
    for row in 0..mat_a_row {
        for col in 0..mat_a_col {
            let idx_a = col + row * mat_a_col;
            let idx_res = row + col * mat_a_row;
            res[idx_res] = a[idx_a];
        }
    }
    res
}
