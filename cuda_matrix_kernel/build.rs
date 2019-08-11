use ptx_builder::error::Result;
use ptx_builder::prelude::*;

fn main() -> Result<()> {
    println!("KERNEL_PTX_PATH = {}", env!("KERNEL_PTX_PATH"));
    CargoAdapter::with_env_var("KERNEL_PTX_PATH").build(Builder::new(".")?)
}
