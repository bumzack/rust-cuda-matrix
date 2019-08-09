#[macro_use]
extern crate rustacuda;
extern crate image;

use image::{ImageBuffer, RgbImage};

use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use rustacuda::memory::DeviceBox;
use std::borrow::Borrow;
use rustacuda::device::DeviceAttribute;

fn show_device_props(device: &Device) {
    println!("devicename: {}", device.name().unwrap());
    println!("total_memory: {}", device.total_memory().unwrap());
    println!("clockrate: {}", device.get_attribute(DeviceAttribute::ClockRate).unwrap());
    println!("ConcurrentKernels: {}", device.get_attribute(DeviceAttribute::ConcurrentKernels).unwrap());
    println!("GlobalMemoryBusWidth: {}", device.get_attribute(DeviceAttribute::GlobalMemoryBusWidth).unwrap());
    println!("L2CacheSize: {}", device.get_attribute(DeviceAttribute::L2CacheSize).unwrap());
    println!("MaxBlockDimX: {}", device.get_attribute(DeviceAttribute::MaxBlockDimX).unwrap());
    println!("MaxBlockDimY: {}", device.get_attribute(DeviceAttribute::MaxBlockDimY).unwrap());
    println!("MaxBlockDimZ: {}", device.get_attribute(DeviceAttribute::MaxBlockDimZ).unwrap());
    println!("MaxGridDimX: {}", device.get_attribute(DeviceAttribute::MaxGridDimX).unwrap());
    println!("MaxGridDimY: {}", device.get_attribute(DeviceAttribute::MaxGridDimY).unwrap());
    println!("MaxGridDimZ: {}", device.get_attribute(DeviceAttribute::MaxGridDimZ).unwrap());
    println!("MaxThreadsPerBlock: {}", device.get_attribute(DeviceAttribute::MaxThreadsPerBlock).unwrap());
    println!("MaxThreadsPerMultiprocessor: {}", device.get_attribute(DeviceAttribute::MaxThreadsPerMultiprocessor).unwrap());
    println!("MultiprocessorCount: {}", device.get_attribute(DeviceAttribute::MultiprocessorCount).unwrap());
    println!("WarpSize: {}", device.get_attribute(DeviceAttribute::WarpSize).unwrap());
}

fn render_mandelbrot_cuda() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    // Create a context associated to this device
    let context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("/tmp/ptx-builder-0.5/cuda_kernel_mandel/6def2f1805f66bf6/nvptx64-nvidia-cuda/release/cuda_kernel_mandel.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut width = DeviceBox::new(&640f32)?;
    let mut height = DeviceBox::new(&480f32)?;

    // Allocate space on the device and copy numbers to it.
    let mut pixels = DeviceBuffer::from_slice(&[0f32; 640 * 480 * 3])?;

    // Launching kernels is unsafe since Rust can't enforce safety - think of kernel launches
    // as a foreign-function call. In this case, it is - this kernel is written in CUDA C.

    // pub unsafe extern "ptx-kernel" fn calc_mandel(pixels: *const Pixel, w: usize, h: usize) {
    unsafe {
        // Launch the `add` function with one block containing four threads on the stream.
        launch!(module.calc_mandel<<<2, 3, 0, stream>>>(
            pixels.as_device_ptr(),
           width.as_device_ptr(),
            height.as_device_ptr()
        ))?;
    }

    // The kernel launch is asynchronous, so we wait for the kernel to finish executing
    stream.synchronize()?;

    // Copy the result back to the host
    let mut result_host: Vec<f32> = Vec::new();


    // TODO: copy data from device to host and convert f32 to u8 ?
    // pixels.copy_to(&mut result_host)?;
    // println!("bla sum is {:?}", result_host[100 * 3]);


    let mut i = 0;
    let mut idx: usize = 0;
    let chunk_size = 32;
    // let mut iter = pixels.chunks(chunk_size);
    let mut host_buf = vec![0f32; chunk_size];
    // let w = width.borrow() as i32;
    // let d = pixels.as_mut_ptr();
    for i in pixels.chunks_mut(chunk_size) {
        if i.len() == host_buf.len() {
            i.copy_from(&host_buf).unwrap();
            result_host.append(&mut host_buf);
        } else {
            host_buf = vec![0f32; i.len()];
            i.copy_from(&host_buf).unwrap();
            result_host.append(&mut host_buf);
        }
    }


    // let w1: f32 = width.into();
    let mut w1 = 0.0f32;
    width.copy_to(&mut w1)?;
    let mut h1 = 0.0f32;
    height.copy_to(&mut h1)?;

    let w = w1 as u32;
    let h = h1 as u32;
    let mut image: RgbImage = ImageBuffer::new(w as u32, h as u32);

    let mut x = 0;
    let mut y = 0;
    // println!("  w = {}, h = {}, result_host.len() = {}", w, h, result_host.len());


    for i in 0..result_host.len() / 3 {
        // println!("  x = {}, y = {}, idx = {}, i = {}", x, y, idx, i);

        let pixel = image::Rgb([result_host[idx] as u8, result_host[idx + 1] as u8, result_host[idx + 2] as u8]);
        image.put_pixel(x as u32, y as u32, pixel);
        idx = idx + 3;
        x = x + 1;
        if x % w == 0 {
            x = 0;
            y = y + 1;
        }
    }

    image.save("fractal_cuda.png").unwrap();

    show_device_props(&device);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    render_mandelbrot_cuda()?;
    Ok(())
}
