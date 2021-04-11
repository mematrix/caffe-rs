use cblas::{saxpy, daxpy, sasum, dasum};
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

    pub fn caffe_cpu_asum(n: i32, x: &[f32]) -> f32 {
        unsafe { sasum(n, x, 1) }
    }
}

impl Blas<f64> {
    pub fn caffe_axpy(n: i32, alpha: f64, x: &[f64], y: &mut [f64]) {
        unsafe { daxpy(n, alpha, x, 1, y, 1); }
    }

    pub fn caffe_cpu_asum(n: i32, x: &[f64]) -> f64 {
        unsafe { dasum(n, x, 1) }
    }
}
