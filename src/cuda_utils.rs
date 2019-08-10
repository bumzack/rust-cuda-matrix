use rustacuda::device::DeviceAttribute;
use rustacuda::prelude::*;

pub fn show_device_props(device: &Device) {
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
