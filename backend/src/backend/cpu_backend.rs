use crate::backend::backend::Backend;
use crate::cpu_matrix::cpu_matrix::CpuMatrix;
use crate::matrix::matrix::Matrix;
use std::ops::Add;

pub struct CpuBackend {
    // hmmm ...
}

impl Backend<CpuMatrix, CpuMatrix> for CpuBackend {
    fn add(&self, a: &CpuMatrix, b: &CpuMatrix) -> CpuMatrix {
        a + b
    }

    fn invert(&self, a: &CpuMatrix, b: &CpuMatrix) -> CpuMatrix {
        if a.get_rows() !=b.get_cols() {
            // TODO 
            // return Err(MathError::MatrixNotInvertableNotSquare);
            panic!("matrix dimensions dont fit");
        }
        let rows = a.get_rows();
        let mut cols = a.get_cols();

        // helper matrix for inverting
        let mut dummy = self.new_matrix(rows, 2 * cols);

        // copy matrix a to dummy (left half of dummy)
        for row in 0..rows {
            for col in 0..cols {
                dummy.set(row, col, a.get(row, col));
            }
        }
        // set identiy matrix elements
        for row in 0..rows {
            dummy.set(row, cols + row, 1.0);
        }
        // apply all transformations to the identity matrix as well
        cols = 2 * a.get_cols();

        let mut tmp: f32 = 0.0;
        for row in 0..rows {
            // transform to an upper triangle matrix
            // element in main diagonal is used to divide the current row
            let mut elem = dummy.get(row, row);
            if elem != 0.0 {
                // divide each element in current row by elem -> so A(row,row) = 1
                for col in 0..cols {
                    tmp = dummy.get(row, col);
                    dummy.set(row, col, tmp / elem);
                }
                // subtract the line row from all the rows below this row
                for row2 in row + 1..rows {
                    elem = dummy.get(row2, row);
                    for col in 0..cols {
                        // dummy.data[row2][col] = dummy.data[row2][col] - elem * dummy.data[row][col];
                        tmp = dummy.get(row2, col) - elem * dummy.get(row, col);
                        dummy.set(row2, col, tmp);
                    }
                }
            } else {
                // TODO 
                // return Err(MathError::MatrixNotInvertable(row, row, elem as f64));
                panic!("matrix is not invertible dont fit");
            }
        }

        // all elements below the main diagonal are 0
        // iterate from the last row to the first row and
        // set the elements right from the diagonal to 0
        // transform to an upper triangle matri
        // element in main diagonal is used to divide the current row

        for row in (1..rows).rev() {
            // transform to an lower triangle matrix
            // subtract the line row from all the rows above  this row

            for row2 in (0..row).rev() {
                let elem = dummy.get(row2, row);
                for col in 0..cols {
                    // dummy.data[row2][col] = dummy.data[row2][col] - elem * dummy.data[row][col];
                    tmp = dummy.get(row2, col) - elem * dummy.get(row, col);
                    dummy.set(row2, col, tmp);
                }
            }
        }

        let mut res = self.new_matrix(rows, rows);
        for row in 0..rows {
            for col in 0..a.get_cols() {
                // res.data[row][col] = dummy.data[row][col + a.cols];
                tmp = dummy.get(row, col + a.get_cols());
                res.set(row, col, tmp);
            }
        }
        res
    }

    fn mul(&self, a: &CpuMatrix, b: &CpuMatrix) -> CpuMatrix {
        let mut res = CpuMatrix::zero(a.get_rows(), a.get_cols());
        for r in 0..a.get_rows() {
            for c in 0..a.get_cols() {
                let mut tmp = 0f32;
                for i in 0..a.get_cols() {
                    tmp = tmp + a.get(r, i) * b.get(i, c);
                }
                res.set(r, c,  tmp);
            }
        }
        res
    }

    fn new_matrix(&self, rows: usize, cols: usize) -> CpuMatrix {
        CpuMatrix::zero(rows, cols)
    }
}

impl CpuBackend {
    pub fn new() -> CpuBackend {
        CpuBackend {}
    }
}

impl <'a, 'b>  Add<&'b CpuMatrix> for &'a CpuMatrix {
    type Output = CpuMatrix;

    fn add(self, other: &'b CpuMatrix) -> CpuMatrix {
        let mut res = CpuMatrix::zero(self.get_rows(), self.get_cols());
        for r in 0..self.get_rows() {
            for c in 0..self.get_cols() {
                res.set(r, c,  self.get(r, c) + other.get(r, c));
            }
        }
        res
    }
}

//
//impl<'a, 'b> Mul<&'b Tuple4D> for &'a Matrix {
//    type Output = Tuple4D;
//
//    fn mul(self, rhs: &'b Tuple4D) -> Tuple4D {
//        assert!(self.rows == 4);
//        let mut t = Tuple4D::empty();
//
//        // TODO: not a generic code for general matrix dimensions
//        t.x = self.m[0][0] * rhs.x + self.m[0][1] * rhs.y + self.m[0][2] * rhs.z + self.m[0][3] * rhs.w;
//        t.y = self.m[1][0] * rhs.x + self.m[1][1] * rhs.y + self.m[1][2] * rhs.z + self.m[1][3] * rhs.w;
//        t.z = self.m[2][0] * rhs.x + self.m[2][1] * rhs.y + self.m[2][2] * rhs.z + self.m[2][3] * rhs.w;
//        t.w = self.m[3][0] * rhs.x + self.m[3][1] * rhs.y + self.m[3][2] * rhs.z + self.m[3][3] * rhs.w;
//
//        t
//    }
//}
