use std::fmt;
use crate::matrix::matrix::Matrix;


#[derive(Debug)]
pub struct CpuMatrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl fmt::Display for CpuMatrix {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "\nCPU Matrix \nrows: {}, cols: {}\n",
            self.get_rows(), self.get_cols()
        )?;
        for row in 0..self.get_rows() {
            for col in 0..self.get_cols() {
                write!(f, " {} ", self.get(row, col))?;
            }
            write!(f, "\n ")?;
        }
        write!(f, "\n ")
    }
}

impl Matrix<CpuMatrix> for CpuMatrix {
    fn one(rows: usize, cols: usize) -> CpuMatrix {
        CpuMatrix {
            rows: rows,
            cols: cols,
            data: vec![1f32; rows * cols],
        }
    }

    fn zero(rows: usize, cols: usize) -> CpuMatrix {
        CpuMatrix {
            rows: rows,
            cols: cols,
            data: vec![0f32; rows * cols],
        }
    }

    fn identity(rows: usize) -> CpuMatrix {
        let mut m = CpuMatrix::zero(rows, rows);
        for i in 0..rows {
            m.data[i * rows + i] = 1.0;
        }
        m
    }

    fn from(data: Vec<f32>, rows: usize, cols: usize) -> CpuMatrix {
        assert_eq!(data.len(), rows * cols);
        CpuMatrix {
            rows: rows,
            cols: cols,
            data: data,
        }
    }

    fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    fn get_rows(&self) -> usize {
        self.rows
    }

    fn get_cols(&self) -> usize {
        self.cols
    }
}

