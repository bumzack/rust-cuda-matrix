use std::error::Error;
use std::ffi::CString;
use std::time::Instant;
use crate::PTX_CODE;

use rustacuda::prelude::*;

pub fn add_matrix_2D2D() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    let nx = 8000;
    let ny = 5000;
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
    let module_data = CString::new(PTX_CODE)?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let b = (32, 32, 1);
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
        let res = launch!(module.sumMatrixOnGpu2D2D<<<grid, block, 0, stream>>>(
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
    println!("duration gpu  sumMatrixOnGpu2D2D: {:?}", duration_cuda);

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
    // The kernel launch is asynchronous, so we wait for the kernel to finish executing

    // show_device_props(&device);

    Ok(())
}

pub fn add_matrix_2D1D() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    let nx = 8000;
    let ny = 5000;
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
    let module_data = CString::new(PTX_CODE)?;
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

pub fn add_matrix_cpu(a: &Vec<f32>, b: &Vec<f32>, nx: usize, ny: usize) -> Vec<f32> {
    let mut res: Vec<f32> = vec![0f32; nx * ny];
    for x in 0..nx {
        for y in 0..ny {
            res[y * nx + x] = a[y * nx + x] + b[y * nx + x];
        }
    }
    res
}
