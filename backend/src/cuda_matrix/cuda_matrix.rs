use crate::cpu_matrix::cpu_matrix::CpuMatrix;
use crate::matrix::matrix::Matrix;
use rustacuda::prelude::*;
use std::fmt;

#[derive(Debug)]
pub struct CudaMatrix {
    cpu_matrix: CpuMatrix,
    device_buffer: DeviceBuffer<f32>,
}

impl Matrix<CudaMatrix> for CudaMatrix {
    fn one(rows: usize, cols: usize) -> CudaMatrix {
        let m = CpuMatrix::one(rows, cols);
        let d =
            DeviceBuffer::from_slice(&m.get_data()).expect("can't create DeviceBuffer for CudaMatrix::one");
        CudaMatrix {
            cpu_matrix: m,
            device_buffer: d,
        }
    }

    fn zero(rows: usize, cols: usize) -> CudaMatrix {
        let m = CpuMatrix::zero(rows, cols);
        let d =
            DeviceBuffer::from_slice(&m.get_data()).expect("can't create DeviceBuffer for CudaMatrix::zero");
        CudaMatrix {
            cpu_matrix: m,
            device_buffer: d,
        }
    }

    fn identity(rows: usize) -> CudaMatrix {
        let m = CpuMatrix::identity(rows);
        let d = DeviceBuffer::from_slice(&m.get_data())
            .expect("can't create DeviceBuffer for CudaMatrix::identity");
        CudaMatrix {
            cpu_matrix: m,
            device_buffer: d,
        }
    }

    fn from_vec(data: Vec<f32>, rows: usize, cols: usize) -> CudaMatrix {
        let m = CpuMatrix::from_vec(data, rows, cols);
        let d =
            DeviceBuffer::from_slice(&m.get_data()).expect("can't create DeviceBuffer for CudaMatrix::from");
        CudaMatrix {
            cpu_matrix: m,
            device_buffer: d,
        }
    }

    fn set(&mut self, row: usize, col: usize, value: f32) {

        self.cpu_matrix.set(row, col, value);
        // TODO: is that ok? every time a value is modified, we create a new DeviceBuffer?
        self.device_buffer = DeviceBuffer::from_slice(&self.get_cpu_matrix().get_data())
            .expect("can't create DeviceBuffer for CudaMatrix::set");
    }

    fn get(&self, row: usize, col: usize) -> f32 {
        self.get_cpu_matrix().get(row, col)
    }

    fn get_rows(&self) -> usize {
        self.cpu_matrix.get_rows()
    }

    fn get_cols(&self) -> usize {
        self.cpu_matrix.get_cols()
    }
}

impl CudaMatrix {
  pub  fn get_device_buffer(&self) -> &DeviceBuffer<f32> {
        &self.device_buffer
    }

    pub  fn get_device_buffer_mut (&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.device_buffer
    }

    pub fn set_device_matrix(&mut self, d: DeviceBuffer<f32>) {
        self.device_buffer = d;
    }

    pub fn get_cpu_matrix(&self) -> &CpuMatrix {
        &self.cpu_matrix
    }

    pub fn copy_from_device_to_host(&mut self) {
        self.device_buffer.copy_to(self.cpu_matrix.get_data_mut());
    }
}

// TODO: what do we want to display here ...
impl fmt::Display for CudaMatrix {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "\nthe CPU matrix of the CUDA matrix contains: \nrows: {}, cols: {}\n",
            self.cpu_matrix.get_rows(),
            self.cpu_matrix.get_cols()
        )?;
        for row in 0..self.cpu_matrix.get_rows() {
            for col in 0..self.cpu_matrix.get_cols() {
                write!(f, " {} ", self.cpu_matrix.get(row, col))?;
            }
            write!(f, "\n ")?;
        }
        write!(f, "\n ")
    }
}
