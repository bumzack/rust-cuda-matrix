#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", feature(stdsimd, proc_macro_hygiene, abi_ptx))]

// #![cfg_attr(target_os = "cuda", feature(abi_ptx, proc_macro_hygiene, core_intrinsics, stdsimd))]

// TODO.md: this is not found? is supposed to be not found, because target-os = cuda, or should this work?
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
pub unsafe extern "ptx-kernel" fn sumMatrixOnGpu2D2D(
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
pub unsafe extern "ptx-kernel" fn sumMatrixOnGpu2D1D(
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
    let iy = Context::block().index().y as isize;
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
pub unsafe extern "ptx-kernel" fn matrix_mul_2D2D(
    mat_a: *const f32,
    mat_b: *const f32,
    mat_c: *mut f32,
    mat_a_row: usize,
    mat_a_col: usize,
    mat_b_row: usize,
    mat_b_col: usize,
    block_dim_x: u32,
    block_dim_y: u32,
    // TODO.md: add succes boolean as return value
    // success: *mut bool,
) {
    use ptx_support::prelude::*;

    if mat_a_col != mat_b_row {
        //     *success = false;
        return;
    }
    //    *success = true;

    let col =
        (Context::thread().index().x + Context::block().index().x * block_dim_x as u64) as isize;
    let row =
        (Context::thread().index().y + Context::block().index().y * block_dim_y as u64) as isize;
    let idx = row * mat_b_col as isize + col;
    // row * mat_b_col + col
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
            let idx_b = col + i as isize * mat_b_col as isize;

            //   let idx_a = i + row * mat_a_col;
            //   let idx_b = i * mat_b_col + col;

            //            if col == 3 && row == 4 {
            //                cuda_printf!(
            //                "XXXXXX col = %ul,  row = %ul,  idx_a = %ul,   idx_b = %ul  \n",
            //                col as u32,
            //                row as u32,
            //                idx_a as u32,
            //                idx_b as u32
            //            );
            //       }

            tmp = tmp + *mat_a.offset(idx_a) * *mat_b.offset(idx_b);
        }
        *mat_c.offset(idx) = tmp;
    }

    //    cuda_printf!("thread_id (%ul, %ul)   block_id (%ul, %ul)   coordinate (%ul, %ul)    global_idx %ul,    a =  %f, b = %f, c = %f\n",
    //            Context::thread().index().x as u32, Context::thread().index().y as u32,
    //            Context::block().index().x as u32 ,  Context::block().index().y as u32,
    //            ix as u32, iy as u32, idx as u32, *mat_a.offset(idx as isize) as f64,  *mat_b.offset(idx as isize) as f64,  *mat_c.offset(idx as isize) as f64);
}

#[no_mangle]
#[cfg(target_os = "cuda")]
pub unsafe extern "ptx-kernel" fn matrix_mul_2D1D(
    mat_a: *const f32,
    mat_b: *const f32,
    mat_c: *mut f32,
    mat_a_row: usize,
    mat_a_col: usize,
    mat_b_row: usize,
    mat_b_col: usize,
    block_dim_x: u32,
    block_dim_y: u32,
    // TODO.md: add succes boolean as reutn value
    // success: *mut bool,
) {
    use ptx_support::prelude::*;

    if mat_a_col != mat_b_row {
        //     *success = false;
        return;
    }
    //    *success = true;

    let col =
        (Context::thread().index().x + Context::block().index().x * block_dim_x as u64) as isize;
    let row = Context::block().index().y as isize;
    let idx = row * mat_b_col as isize + col;
    // row * mat_b_col + col
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
            let idx_b = col + i as isize * mat_b_col as isize;

            //   let idx_a = i + row * mat_a_col;
            //   let idx_b = i * mat_b_col + col;

            //            if col == 3 && row == 4 {
            //                cuda_printf!(
            //                "XXXXXX col = %ul,  row = %ul,  idx_a = %ul,   idx_b = %ul  \n",
            //                col as u32,
            //                row as u32,
            //                idx_a as u32,
            //                idx_b as u32
            //            );
            //       }

            tmp = tmp + *mat_a.offset(idx_a) * *mat_b.offset(idx_b);
        }
        *mat_c.offset(idx) = tmp;
    }

    //    cuda_printf!("thread_id (%ul, %ul)   block_id (%ul, %ul)   coordinate (%ul, %ul)    global_idx %ul,    a =  %f, b = %f, c = %f\n",
    //            Context::thread().index().x as u32, Context::thread().index().y as u32,
    //            Context::block().index().x as u32 ,  Context::block().index().y as u32,
    //            ix as u32, iy as u32, idx as u32, *mat_a.offset(idx as isize) as f64,  *mat_b.offset(idx as isize) as f64,  *mat_c.offset(idx as isize) as f64);
}

#[no_mangle]
#[cfg(target_os = "cuda")]
pub unsafe extern "ptx-kernel" fn transpose_matrix_row_2D2D(
    mat_a: *const f32,
    mat_b: *mut f32,
    mat_a_col: usize,
    mat_a_row: usize,
    block_dim_x: u32,
    block_dim_y: u32,
    // TODO.md: add succes boolean as reutn value
    // success: *mut bool,
) {
    use ptx_support::prelude::*;

    let col =
        (Context::thread().index().x + Context::block().index().x * block_dim_x as u64) as isize;
    let row =
        (Context::thread().index().y + Context::block().index().y * block_dim_y as u64) as isize;
    let idx_a = row * mat_a_col as isize + col;
    let idx_b = col * mat_a_row as isize + row;
    if col < mat_a_col as isize && row < mat_a_row as isize {
        *mat_b.offset(idx_b) = *mat_a.offset(idx_a);
    }
}

#[no_mangle]
#[cfg(target_os = "cuda")]
pub unsafe extern "ptx-kernel" fn transpose_matrix_col_2D2D(
    mat_a: *const f32,
    mat_b: *mut f32,
    mat_a_col: usize,
    mat_a_row: usize,
    block_dim_x: u32,
    block_dim_y: u32,
    // TODO.md: add succes boolean as reutn value
    // success: *mut bool,
) {
    use ptx_support::prelude::*;

    let col =
        (Context::thread().index().x + Context::block().index().x * block_dim_x as u64) as isize;
    let row =
        (Context::thread().index().y + Context::block().index().y * block_dim_y as u64) as isize;
    let idx_a = col * mat_a_row as isize + row;
    let idx_b = row * mat_a_col as isize + col;
    if col < mat_a_col as isize && row < mat_a_row as isize {
        *mat_b.offset(idx_b) = *mat_a.offset(idx_a);
    }
}

// TODO.md: does not work with non-quadratic matrices :-(
// works for 64x64, 128x128, bzt 128x64
//
//#[no_mangle]
//#[cfg(target_os = "cuda")]
//pub unsafe extern "ptx-kernel" fn transpose_matrix_unroll4C(
//    mat_a: *const f32,
//    mat_b: *mut f32,
//    nx: usize,
//    ny: usize,
//    block_dim_x: u32,
//    block_dim_y: u32,
//    // TODO.md: add succes boolean as reutn value
//    // success: *mut bool,
//) {
//    use ptx_support::prelude::*;
//
//    let nx = nx as isize;
//    let ny = ny as isize;
//    let block_dim_x = block_dim_x as isize;
//    let block_dim_y = block_dim_y as isize;
//    let ix = (Context::thread().index().x + Context::block().index().x * block_dim_x as u64 * 4)
//        as isize;
//    let iy =
//        (Context::thread().index().y + Context::block().index().y * block_dim_y as u64) as isize;
//
//    let ti = iy * nx + ix;
//    let to = ix * ny + iy;
//    if (ix + 3 * block_dim_x) < nx && iy < ny {
//        *mat_b.offset(ti) = *mat_a.offset(to);
//        *mat_b.offset(ti + block_dim_x) = *mat_a.offset(to + block_dim_x * ny);
//        *mat_b.offset(ti + 2 * block_dim_x) = *mat_a.offset(to + 2 * block_dim_x * ny);
//        *mat_b.offset(ti + 3 * block_dim_x) = *mat_a.offset(to + 3 * block_dim_x * ny);
//    }
//}


fn main() {}
