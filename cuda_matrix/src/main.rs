#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", feature(stdsimd, proc_macro_hygiene, abi_ptx))]

// #![cfg_attr(target_os = "cuda", feature(abi_ptx, proc_macro_hygiene, core_intrinsics, stdsimd))]

// TODO: this is not found? is supposed to be not found, because target-os = cuda, or should this work?
// use core::arch::nvptx::*;

#[no_mangle]
#[cfg(target_os = "cuda")]
pub unsafe extern "ptx-kernel" fn printThreadIndex(
    values: *mut f32,
    nx: *const usize,
    ny: *const usize,
    block_dim_x: *const usize,
    block_dim_y: *const usize,
) {
    use ptx_support::prelude::*;

//    cuda_printf!(
//        "printThreadIndex   Hello from block(%lu,%lu,%lu) and thread(%lu,%lu,%lu)\n",
//        Context::block().index().x,
//        Context::block().index().y,
//        Context::block().index().z,
//        Context::thread().index().x,
//        Context::thread().index().y,
//        Context::thread().index().z,
//    );
    cuda_printf!("start 56 \n");

    let ix = Context::thread().index().x + Context::block().index().x * block_dim_x as u64;
    let iy = Context::thread().index().y + Context::block().index().y * block_dim_y as u64;
    let idx = iy * *nx as u64 + ix;
    cuda_printf!("thread_id (%ul, %ul)   block_id (%ul, %ul)   coordinate (%ul, %ul)    global_idx %ul,    ival:   %f\n",
            Context::thread().index().x as u32, Context::thread().index().y as u32,
            Context::block().index().x as u32 ,  Context::block().index().y as u32,
            ix as u32, iy as u32, idx as u32, *values.offset(idx as isize) as f64);
}
fn main() {}
