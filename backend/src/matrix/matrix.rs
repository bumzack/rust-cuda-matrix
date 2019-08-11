pub trait Matrix <T> {
    fn one(rows: usize, cols: usize) -> T;
    fn zero(rows: usize, cols: usize) -> T;
    fn identity(rows: usize) -> T;
    fn from(data:  Vec<f32>, rows: usize, cols: usize) -> T;
    // fn random(rows: usize, cols: usize, min: f32, max: f32) -> T;
    fn set(&mut self, row: usize, col: usize, value: f32);
    fn get(&self, row: usize, col: usize) -> f32;
    fn get_rows(&self) -> usize;
    fn get_cols(&self) -> usize;
}
