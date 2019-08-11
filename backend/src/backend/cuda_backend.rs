use crate::backend::backend::Backend;
use crate::cuda_matrix::cuda_matrix::CudaMatrix;
use crate::matrix::matrix::Matrix;
use std::ops::Add;
use rustacuda::prelude::*;
use rustacuda::device::DeviceAttribute;

 use std::ffi::CString;


pub struct CudaBackend {
//    devicename:String,
//    total_memory: usize,
//    clockrate: i32,
//    device: Device,
    context: Context,
//    module: Module,
}

impl Backend<CudaMatrix, CudaMatrix> for CudaBackend {
    fn add(&self, a: & mut CudaMatrix, b: &mut CudaMatrix) -> CudaMatrix {

        // TODO: move this to the CudaBackend::new ??
        let device = Device::get_device(0).expect("can't get CUDA device name 'new'");
        let context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).expect("can't create CUDA context in 'new'");;
        let module_data = CString::new(include_str!(env!("KERNEL_PTX_PATH"))).expect("can't load PTX file in  'new'");;
        let module = Module::load_from_string(&module_data).expect("can't load CUDA modules  'new'");;
        // END todo


        let mut res = self.new_matrix(a.get_rows(), a.get_cols());
        // h_res.set_device_matrix(DeviceBuffer::from_slice(&h_res)?);

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("create new CUDA stream failed in 'add'");

        let bl = (256, 1, 1);
        let block = (bl.0 as u32, bl.1 as u32, bl.2 as u32);

        let g = (
            (a.get_cols() as i32 + block.0 as i32 - 1) / block.0 as i32,
            (a.get_rows() as i32 + block.1 as i32 - 1) / block.1 as i32,
            1 as i32,
        );
        let grid = (g.0 as u32, g.1 as u32, 1 as u32);

        println!("block = {:?}, grid = {:?}", block, grid);


        unsafe {
            // Launch the `add` function with one block containing four threads on the stream.
            let res = launch!(module.sumMatrixOnGpu2D1D<<<grid, block, 0, stream>>>(
            a.get_device_buffer_mut().as_device_ptr(),
            b.get_device_buffer_mut().as_device_ptr(),
            res.get_device_buffer_mut().as_device_ptr(),
            a.get_cols(),
            a.get_rows(),
            block.0,
            block.1
        )).expect("add two matrices crashed");
        }
        stream.synchronize().expect("can't sychronize CUDA stream in 'add'");
        res
    }

    fn invert(&self, a: &CudaMatrix, b: &CudaMatrix) -> CudaMatrix {
        CudaMatrix::zero(a.get_rows(), a.get_cols())
    }

    fn mul(&self, a: &CudaMatrix, b: &CudaMatrix) -> CudaMatrix {
        CudaMatrix::zero(a.get_rows(), a.get_cols())
    }

    fn new_matrix(&self, rows: usize, cols: usize) -> CudaMatrix {

        CudaMatrix::zero(rows, cols)
    }
}

impl CudaBackend {
    pub fn new() -> CudaBackend {

        rustacuda::init(CudaFlags::empty()).expect("can't initialize CUDA in 'new'");
        let device = Device::get_device(0).expect("can't get CUDA device name 'new'");
        let context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).expect("can't create CUDA context in 'new'");;

//        // TODO: do this here once and not on every method call?
//        rustacuda::init(CudaFlags::empty()).expect("can't initialize CUDA in 'new'");
//        let device = Device::get_device(0).expect("can't get CUDA device name 'new'");
//        let context =
//            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).expect("can't create CUDA context in 'new'");;
//        let module_data = CString::new(include_str!(env!("KERNEL_PTX_PATH"))).expect("can't load PTX file in  'new'");;
//        let module = Module::load_from_string(&module_data).expect("can't load CUDA modules  'new'");;
//        // END todo


        CudaBackend {
//            devicename: device.name().unwrap() ,
//            total_memory: device.total_memory().unwrap(),
//            clockrate: device.get_attribute(DeviceAttribute::ClockRate).unwrap(),
//            device: device,
            context: context,
//            module: module,
        }
    }
}

impl<'a, 'b> Add<&'b CudaMatrix> for &'a CudaMatrix {
    type Output = CudaMatrix;

    fn add(self, other: &'b CudaMatrix) -> CudaMatrix {
        let mut res = CudaMatrix::zero(self.get_rows(), self.get_cols());
        for r in 0..self.get_rows() {
            for c in 0..self.get_cols() {
                res.set(r, c, self.get(r, c) + other.get(r, c));
            }
        }
        // always copy the data from the device to the host after a computation
        res.copy_from_device_to_host();

        res
    }
}



