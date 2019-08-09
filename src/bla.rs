extern crate image;

use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

use image::{ImageBuffer, RgbImage};

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

fn main() {
    const WIDTH: usize = 1920;
    const HEIGHT: usize = 1080;

    let width = WIDTH as f32;
    let height = HEIGHT as f32;

    let start = Instant::now();
    println!("start: {:?}", start);

    let mut image: RgbImage = ImageBuffer::new(WIDTH as u32, HEIGHT as u32);

    let num_threads = 4;

    let data = Arc::new(Mutex::new(image));

    let mut children = vec![];

    let mut act_y = 0;

    for i in 0..num_threads {
        let cloned_data = data.clone();

        children.push(thread::spawn(move || {
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

            let inc_real = (right_top_real - left_bottom_real) / width;
            let inc_img = (right_top_img - left_bottom_img) / height;

            let mut y = 0;

            while act_y < HEIGHT {
                if y < HEIGHT {
                    y = act_y;
                    act_y += 1;
                }

                // println!("thread {}, y = {}, act_Y = {}", i, y, act_y);

                for x in 0..WIDTH {
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
                    let pixel = image::Rgb([red as u8, green as u8, blue as u8]);

                    let mut img = cloned_data.lock().unwrap();
                    img.put_pixel(x as u32, y as u32, pixel);
                }
            }
        }));
    }
    for child in children {
        // Wait for the thread to finish. Returns a result.
        let _ = child.join();
    }

    let stopped = Instant::now();
    println!("end: {:?}", stopped);
    println!("{:?}", stopped.duration_since(start));

    // image.save("fractal.png").unwrap();
}
