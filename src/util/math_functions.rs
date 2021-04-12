use cblas::{saxpy, daxpy, sasum, dasum, sdot, ddot, sscal, dscal};
use std::marker::PhantomData;


pub struct CaffeUtil<T: Sized> {
    _phantom: PhantomData<T>,
}

impl<T: Sized + Copy> CaffeUtil<T> {
    pub fn caffe_copy(n: usize, x: &[T], y: &mut [T]) {
        if x.as_ptr() != y.as_ptr() {
            debug_assert!(x.len() >= n && y.len() >= n);

            y[..n].copy_from_slice(&x[..n]);
        }
    }
}

pub struct Blas<T: Sized> {
    _phantom: PhantomData<T>,
}

impl Blas<f32> {
    pub fn caffe_axpy(n: i32, alpha: f32, x: &[f32], y: &mut [f32]) {
        unsafe { saxpy(n, alpha, x, 1, y, 1); }
    }

    pub fn caffe_cpu_asum(n: i32, x: &[f32]) -> f32 {
        unsafe { sasum(n, x, 1) }
    }

    pub fn caffe_cpu_strided_dot(n: i32, x: &[f32], inc_x: i32, y: &[f32], inc_y: i32) -> f32 {
        unsafe { sdot(n, x, inc_x, y, inc_y) }
    }

    pub fn caffe_cpu_dot(n: i32, x: &[f32], y: &[f32]) -> f32 {
        Self::caffe_cpu_strided_dot(n, x, 1, y, 1)
    }

    pub fn caffe_scal(n: i32, alpha: f32, x: &mut [f32]) {
        unsafe { sscal(n, alpha, x, 1) }
    }
}

impl Blas<f64> {
    pub fn caffe_axpy(n: i32, alpha: f64, x: &[f64], y: &mut [f64]) {
        unsafe { daxpy(n, alpha, x, 1, y, 1); }
    }

    pub fn caffe_cpu_asum(n: i32, x: &[f64]) -> f64 {
        unsafe { dasum(n, x, 1) }
    }

    pub fn caffe_cpu_strided_dot(n: i32, x: &[f64], inc_x: i32, y: &[f64], inc_y: i32) -> f64 {
        unsafe { ddot(n, x, inc_x, y, inc_y) }
    }

    pub fn caffe_cpu_dot(n: i32, x: &[f64], y: &[f64]) -> f64 {
        Self::caffe_cpu_strided_dot(n, x, 1, y, 1)
    }

    pub fn caffe_scal(n: i32, alpha: f64, x: &mut [f64]) {
        unsafe { dscal(n, alpha, x, 1) }
    }
}
