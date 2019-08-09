#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "std"), feature(core_intrinsics))]

pub struct Pixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Pixel {
    pub fn new() -> Pixel {
        Pixel {
            r: 0,
            g: 0,
            b: 0,
        }
    }
}
//
//pub struct Image {
//    pub width: usize,
//    pub height: usize,
//    pub pixels: Vec<Pixel>,
//}
//
//impl Image {
//    fn new(width: usize, height: usize) -> Image {
//        Image {
//            width: width,
//            height: height,
//            pixels: vec![Pixel::new(); width * height],
//        }
//    }
//}
//
