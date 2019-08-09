#[macro_use]
extern crate rustacuda;

use rustacuda::device::DeviceAttribute;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

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

    println!("befire max_block_dim_x");
    let max_block_dim_x = device.get_attribute(DeviceAttribute::MaxBlockDimX).unwrap();
    let max_block_dim_y = device.get_attribute(DeviceAttribute::MaxBlockDimY).unwrap();

    let nx = 6;
    let ny = 8;
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
    let context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;


    println!("befire d_matrix");
    let mut d_matrix = DeviceBuffer::from_slice(&matrix)?;
    println!("befire 1");
    let mut d_nx = DeviceBox::new(&(nx)).unwrap();
    println!("befire 2");
    let mut d_ny = DeviceBox::new(&(ny)).unwrap();
    println!("befire 3");
    let mut d_max_block_dim_x = DeviceBox::new(&(max_block_dim_x)).unwrap();
    println!("befire 4");
    let mut d_max_block_dim_y = DeviceBox::new(&(max_block_dim_y)).unwrap();

    println!("befire let context");

    println!("after let context");
    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("/tmp/ptx-builder-0.5/cuda_matrix/9132ac0994b05664/nvptx64-nvidia-cuda/release/cuda_matrix.ptx"))?;
    println!("after let module_data");
    let module = Module::load_from_string(&module_data)?;
    println!("after let module");

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    println!("after stream");
    println!("d_matrix.len = {}", d_matrix.len());

    let b = (4   , 2 , 1);
    let block =(b.0 as u32, b.1 as u32, b.2 as u32 );

    let g = ((nx as i32 + block.0 as i32  - 1) / block.0 as i32 , (ny as i32 + block.1 as i32  - 1) / block.1 as i32  ,1  as i32 ) ;
    let grid =(g.0 as u32, g.1 as u32, g.2 as u32 );


    unsafe {
        // Launch the `add` function with one block containing four threads on the stream.
        let res = launch!(module.printThreadIndex<<<grid, block, 0, stream>>>(
            d_matrix.as_device_ptr(),
            d_nx.as_device_ptr(),
            d_ny.as_device_ptr(),
            d_max_block_dim_x.as_device_ptr(),
            d_max_block_dim_y.as_device_ptr()
        ));

        match res {
            Ok(o) => println!("everything ok"),
            Err(e) => println!("an error occured: {}", e),
        }
    }
    stream.synchronize()?;

    println!("here");
    // The kernel launch is asynchronous, so we wait for the kernel to finish executing

    // show_device_props(&device);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    test_matrix()?;
    Ok(())
}
