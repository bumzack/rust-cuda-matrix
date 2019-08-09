#![cfg_attr(target_os = "cuda", feature(abi_ptx, proc_macro_hygiene))]
#![cfg_attr(target_os = "cuda", no_std)]

fn get_color_from_iterations(i: i32, max_iterations: i32) -> (i32, i32, i32) {
    if i == max_iterations {
        (0, 0, 0)
    } else {
        let mut red = (i % 32) * 3;
        if red > 255 {
            red = 255;
        }

        let mut green = (i % 16) * 2;
        if green > 255 {
            green = 255;
        }

        let mut blue = (i % 128) * 14;
        if blue > 255 {
            blue = 255;
        }
        (red, green, blue)
    }
}


#[no_mangle]
#[cfg(target_os = "cuda")]
pub unsafe extern "ptx-kernel" fn calc_mandel(pixels: *mut f32, width: *const f32, height: *const f32) {
    use ptx_support::prelude::*;

    cuda_printf!(
        "calc_mandel   Hello from block(%lu,%lu,%lu) and thread(%lu,%lu,%lu)\n",
        Context::block().index().x,
        Context::block().index().y,
        Context::block().index().z,
        Context::thread().index().x,
        Context::thread().index().y,
        Context::thread().index().z,
    );

    let w = *width as isize;
    let h = *height as isize;

    let max_iterations = 10000;
    let max_radius = 4.0;

    let mut z_real;
    let mut z_img;
    let mut z_new_real;
    let mut z_new_img;

    let left_bottom_real = -2.0;
    let left_bottom_img = -1.0;
    let right_top_real = 1.0;
    let right_top_img = 1.0;

    let mut c_real;
    let mut c_img;

    let inc_real = (right_top_real - left_bottom_real) / *width;
    let inc_img = (right_top_img - left_bottom_img) / *height;

    let x_idx = Context::thread().index().x as isize;
    let y_idx = Context::thread().index().y as isize;

    let x = x_idx as f32;
    let y = y_idx as f32;

    cuda_printf!("calc_mandel:   x = %f,  y = %f,    \n",  x as f64, y as f64);

    c_real = left_bottom_real + x as f32 * inc_real;
    c_img = left_bottom_img + y as f32 * inc_img;

    z_real = 0.0;
    z_img = 0.0;

    let mut i = 0;

    while z_real * z_real + z_img * z_img < max_radius && i < max_iterations {
        z_new_real = z_real * z_real - z_img * z_img + c_real;
        z_new_img = 2.0 * z_real * z_img + c_img;

        z_real = z_new_real;
        z_img = z_new_img;

        i += 1;
    }

    let (red, green, blue) = get_color_from_iterations(i, max_iterations);

    let idx = y_idx * w * 3 + x_idx;

    cuda_printf!("calc_mandel:  x_idx = %ul, y_idx = %ul, w = %ul,   h = %ul    \n",  x_idx as u32, y_idx as u32, w  as u32, h  as u32);
    // cuda_printf!("calc_mandel:  width = %f,, height = %f   \n",  *width as f64, *height as f64);

    cuda_printf!("calc_mandel:  red = %f, green = %f , blue = %f  \n",  red as f64, green as  f64, blue as f64);

    *pixels.offset(idx) = red as f32;
    *pixels.offset(idx + 1) = green as f32;
    *pixels.offset(idx + 2) = blue as f32;
}

fn main() {}
