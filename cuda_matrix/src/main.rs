#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", feature(stdsimd, proc_macro_hygiene, abi_ptx))]

// #![cfg_attr(target_os = "cuda", feature(abi_ptx, proc_macro_hygiene, core_intrinsics, stdsimd))]

// TODO: this is not found? is supposed to be not found, because target-os = cuda, or should this work?
// use core::arch::nvptx::*;

#[no_mangle]
#[cfg(target_os = "cuda")]
pub unsafe extern "ptx-kernel" fn printThreadIndex(
    values: *mut f32,
    nx: usize,
    ny: usize,
    block_dim_x: u32,
    block_dim_y: u32,
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
    let ix = Context::thread().index().x + Context::block().index().x * block_dim_x as u64;
    let iy = Context::thread().index().y + Context::block().index().y * block_dim_y as u64;
    let idx = iy * nx as u64 + ix;
    cuda_printf!("thread_id (%ul, %ul)   block_id (%ul, %ul)   coordinate (%ul, %ul)    global_idx %ul,    ival:   %f\n",
            Context::thread().index().x as u32, Context::thread().index().y as u32,
            Context::block().index().x as u32 ,  Context::block().index().y as u32,
            ix as u32, iy as u32, idx as u32, *values.offset(idx as isize) as f64);
}

#[no_mangle]
#[cfg(target_os = "cuda")]
pub unsafe extern "ptx-kernel" fn sumMatrixOnGpu2D(
    mat_a: *const f32,
    mat_b: *const f32,
    mat_c: *mut f32,
    nx: usize,
    ny: usize,
    block_dim_x: u32,
    block_dim_y: u32,
) {
    use ptx_support::prelude::*;

    let ix =
        (Context::thread().index().x + Context::block().index().x * block_dim_x as u64) as isize;
    let iy =
        (Context::thread().index().y + Context::block().index().y * block_dim_y as u64) as isize;
    let idx = iy * nx as isize + ix;
    if ix < nx as isize && iy < ny as isize {
        *mat_c.offset(idx) = *mat_a.offset(idx) + *mat_b.offset(idx);
    }

    //    cuda_printf!("thread_id (%ul, %ul)   block_id (%ul, %ul)   coordinate (%ul, %ul)    global_idx %ul,    a =  %f, b = %f, c = %f\n",
    //            Context::thread().index().x as u32, Context::thread().index().y as u32,
    //            Context::block().index().x as u32 ,  Context::block().index().y as u32,
    //            ix as u32, iy as u32, idx as u32, *mat_a.offset(idx as isize) as f64,  *mat_b.offset(idx as isize) as f64,  *mat_c.offset(idx as isize) as f64);
}

#[no_mangle]
#[cfg(target_os = "cuda")]
pub unsafe extern "ptx-kernel" fn matrix_mul(
    mat_a: *const f32,
    mat_b: *const f32,
    mat_c: *mut f32,
    mat_a_row: usize,
    mat_a_col: usize,
    mat_b_row: usize,
    mat_b_col: usize,
    block_dim_x: u32,
    block_dim_y: u32,
    // TODO: add succes boolean as reutn value
    // success: *mut bool,
) {
    use ptx_support::prelude::*;

    if (mat_a_col != mat_b_row) {
        //     *success = false;
        return;
    }
    //    *success = true;

    let col =
        (Context::thread().index().x + Context::block().index().x * block_dim_x as u64) as isize;
    let row =
        (Context::thread().index().y + Context::block().index().y * block_dim_y as u64) as isize;
    let idx = row * mat_a_col as isize + col;

//    cuda_printf!(
//        "YYYYYYYY  col = %ul,  row = %ul,  idx = %ul \n",
//        col as u32,
//        row as u32,
//        idx as u32
//    );

    if col < mat_b_col as isize && row < mat_a_row as isize {
        let mut tmp = 0f32;
        for i in 0..mat_a_col {
            let idx_a = row * mat_a_col as isize + i as isize;
            let idx_b = row + i as isize * mat_b_col as isize;
            cuda_printf!(
                "XXXXXX col = %ul,  row = %ul,  idx_a = %ul,   idx_b = %ul  \n",
                col as u32,
                row as u32,
                idx_a as u32,
                idx_b as u32
            );
            tmp = tmp + *mat_a.offset(idx_a) * *mat_b.offset(idx_b);
        }
        *mat_c.offset(idx) = tmp;
    }

    //    cuda_printf!("thread_id (%ul, %ul)   block_id (%ul, %ul)   coordinate (%ul, %ul)    global_idx %ul,    a =  %f, b = %f, c = %f\n",
    //            Context::thread().index().x as u32, Context::thread().index().y as u32,
    //            Context::block().index().x as u32 ,  Context::block().index().y as u32,
    //            ix as u32, iy as u32, idx as u32, *mat_a.offset(idx as isize) as f64,  *mat_b.offset(idx as isize) as f64,  *mat_c.offset(idx as isize) as f64);
}

fn main() {}
