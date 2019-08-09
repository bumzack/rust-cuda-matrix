#[macro_use]
extern crate rustacuda;

use rustacuda::device::DeviceAttribute;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use std::time::{Duration, Instant};

fn show_device_props(device: &Device) {
    println!("devicename: {}", device.name().unwrap());
    println!("total_memory: {}", device.total_memory().unwrap());
    println!(
        "clockrate: {}",
        device.get_attribute(DeviceAttribute::ClockRate).unwrap()
    );
    println!(
        "ConcurrentKernels: {}",
        device
            .get_attribute(DeviceAttribute::ConcurrentKernels)
            .unwrap()
    );
    println!(
        "GlobalMemoryBusWidth: {}",
        device
            .get_attribute(DeviceAttribute::GlobalMemoryBusWidth)
            .unwrap()
    );
    println!(
        "L2CacheSize: {}",
        device.get_attribute(DeviceAttribute::L2CacheSize).unwrap()
    );
    println!(
        "MaxBlockDimX: {}",
        device.get_attribute(DeviceAttribute::MaxBlockDimX).unwrap()
    );
    println!(
        "MaxBlockDimY: {}",
        device.get_attribute(DeviceAttribute::MaxBlockDimY).unwrap()
    );
    println!(
        "MaxBlockDimZ: {}",
        device.get_attribute(DeviceAttribute::MaxBlockDimZ).unwrap()
    );
    println!(
        "MaxGridDimX: {}",
        device.get_attribute(DeviceAttribute::MaxGridDimX).unwrap()
    );
    println!(
        "MaxGridDimY: {}",
        device.get_attribute(DeviceAttribute::MaxGridDimY).unwrap()
    );
    println!(
        "MaxGridDimZ: {}",
        device.get_attribute(DeviceAttribute::MaxGridDimZ).unwrap()
    );
    println!(
        "MaxThreadsPerBlock: {}",
        device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap()
    );
    println!(
        "MaxThreadsPerMultiprocessor: {}",
        device
            .get_attribute(DeviceAttribute::MaxThreadsPerMultiprocessor)
            .unwrap()
    );
    println!(
        "MultiprocessorCount: {}",
        device
            .get_attribute(DeviceAttribute::MultiprocessorCount)
            .unwrap()
    );
    println!(
        "WarpSize: {}",
        device.get_attribute(DeviceAttribute::WarpSize).unwrap()
    );
}

fn test_matrix() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    show_device_props(&device);

    let nx = 8;
    let ny = 6;
    let nxy = nx * ny;

    let mut matrix = vec![0f32; nxy];
    let mut blupp = 1f32;
    for elem in matrix.iter_mut() {
        *elem = blupp;
        blupp = blupp + 1.0;
    }

    println!("matrix.len = {}", matrix.len());
    println!("matrix = {:?}", matrix);

    // Create a context associated to this device
    let _context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let mut d_matrix = DeviceBuffer::from_slice(&matrix)?;

    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("/tmp/ptx-builder-0.5/cuda_matrix/9132ac0994b05664/nvptx64-nvidia-cuda/release/cuda_matrix.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let b = (4, 2, 1);
    let block = (b.0 as u32, b.1 as u32, b.2 as u32);

    let g = (
        (nx as i32 + block.0 as i32 - 1) / block.0 as i32,
        (ny as i32 + block.1 as i32 - 1) / block.1 as i32,
        1 as i32,
    );
    let grid = (g.0 as u32, g.1 as u32, 1 as u32);

    println!("block = {:?}, grid = {:?}", block, grid);

    unsafe {
        // Launch the `add` function with one block containing four threads on the stream.
        let res = launch!(module.printThreadIndex<<<grid, block, 0, stream>>>(
            d_matrix.as_device_ptr(),
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

    println!("here");
    // The kernel launch is asynchronous, so we wait for the kernel to finish executing

    Ok(())
}

fn add_matrix() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    let nx = 8000;
    let ny = 6000;
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

    let b = (4, 2, 1);
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
        let res = launch!(module.sumMatrixOnGpu2D<<<grid, block, 0, stream>>>(
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

//    println!("out_host  = {:?}", out_host);

    let start_cpu = Instant::now();
    let res_cpu = add_matrix_cpu(&matrix_a, &matrix_b, nx, ny);
    let duration_cpu = start_cpu.elapsed();

    println!("duration gpu: {:?}", duration_cuda);
    println!("duration cpu: {:?}", duration_cpu);

    for x in 0..res_cpu.len() {
        assert_eq!(res_cpu[x], out_host[x]);
    }
    // The kernel launch is asynchronous, so we wait for the kernel to finish executing

    // show_device_props(&device);

    Ok(())
}

fn add_matrix_cpu(a: &Vec<f32>, b: &Vec<f32>, nx: usize, ny: usize) -> Vec<f32> {
    let mut res: Vec<f32> = vec![0f32; nx * ny];
    for x in 0..nx {
        for y in 0..ny {
            res[y * nx + x] = a[y * nx + x] + b[y * nx + x];
        }
    }
    res
}

fn main() -> Result<(), Box<dyn Error>> {
    // test_matrix()?;
    add_matrix()?;
    Ok(())
}
