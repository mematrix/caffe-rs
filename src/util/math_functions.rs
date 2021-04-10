use cblas::{saxpy};
use std::marker::PhantomData;


pub struct Blas<T: Sized> {
    _phantom: PhantomData<T>,
}

// impl<T: Sized> Blas<T> {
//     pub fn caffe_axpy(n: i32, alpha: T, x: &[T], y: &mut [T]) {
//         unimplemented!();
//     }
// }

impl Blas<f32> {
    pub fn caffe_axpy(n: i32, alpha: f32, x: &[f32], y: &mut [f32]) {
        unsafe { saxpy(n, alpha, x, 1, y, 1); }
    }
}
