use std::error::Error;
use std::ffi::CString;
use std::time::Instant;

use crate::cuda_utils::show_device_props;
use rustacuda::device::DeviceAttribute;
use rustacuda::prelude::*;

pub fn test_matrix() -> Result<(), Box<dyn Error>> {
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

pub fn print_matrix(m: &Vec<f32>, rows: usize, cols: usize) {
    for row in 0..rows {
        for col in 0..cols {
            print!("{}  ", m[col + row * cols]);
        }
        println!();
    }
}
