//pub enum BackendTypeEnum {
//    CPU_SINGLE_CORE,
//    CUDA,
//    // CPU_MULTI_CORE
//}
use crate::matrix::matrix::Matrix;

pub trait Backend<T, M: Matrix<T>> {
     fn add(&self, a: &M, b: &M) -> M;
    fn invert(&self, a: &M, b: &M) -> M;
    fn mul(&self, a: &M, b: &M) -> M;

    fn new_matrix(&self, rows: usize, cols: usize) -> M;
}
