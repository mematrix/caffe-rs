use cblas::{saxpy, daxpy, sasum, dasum, sdot, ddot, sscal, dscal};
use std::marker::PhantomData;
use std::ops::{AddAssign, Div};


pub trait CaffeNum : Copy + Sized + AddAssign + Div {
    fn is_zero(&self) -> bool;

    fn from_f64(v: f64) -> Self;

    fn from_f32(v: f32) -> Self;

    fn from_usize(v: usize) -> Self;

    fn from_div(v: <Self as Div<Self>>::Output) -> Self;

    fn sqrt(v: Self) -> Self;
}

impl CaffeNum for i32 {
    fn is_zero(&self) -> bool {
        *self == 0
    }

    fn from_f64(v: f64) -> Self {
        v as i32
    }

    fn from_f32(v: f32) -> Self {
        v as i32
    }

    fn from_usize(v: usize) -> Self {
        v as i32
    }

    fn from_div(v: Self::Output) -> Self {
        v
    }

    fn sqrt(v: Self) -> Self {
        (v as f64).sqrt() as i32
    }
}

impl CaffeNum for f32 {
    fn is_zero(&self) -> bool {
        *self == 0f32
    }

    fn from_f64(v: f64) -> Self {
        v as f32
    }

    fn from_f32(v: f32) -> Self {
        v
    }

    fn from_usize(v: usize) -> Self {
        v as f32
    }

    fn from_div(v: Self::Output) -> Self {
        v
    }

    fn sqrt(v: Self) -> Self {
        v.sqrt()
    }
}

impl CaffeNum for f64 {
    fn is_zero(&self) -> bool {
        *self == 0f64
    }

    fn from_f64(v: f64) -> Self {
        v
    }

    fn from_f32(v: f32) -> Self {
        v as f64
    }

    fn from_usize(v: usize) -> Self {
        v as f64
    }

    fn from_div(v: Self::Output) -> Self {
        v
    }

    fn sqrt(v: Self) -> Self {
        v.sqrt()
    }
}


pub struct CaffeUtil<T: CaffeNum> {
    _phantom: PhantomData<T>,
}

impl<T: CaffeNum> CaffeUtil<T> {
    pub fn caffe_copy(n: usize, x: &[T], y: &mut [T]) {
        if x.as_ptr() != y.as_ptr() {
            debug_assert!(x.len() >= n && y.len() >= n);

            y[..n].copy_from_slice(&x[..n]);
        }
    }

    pub fn caffe_set(n: usize, alpha: T, y: &mut [T]) {
        assert!(y.len() >= n);
        if alpha.is_zero() {
            unsafe { std::ptr::write_bytes(y.as_mut_ptr(), 0u8, n); }
        } else {
            for x in &mut y[0..n] {
                *x = alpha;
            }
        }
    }
}


pub trait BlasOp<T: Sized> {
    fn caffe_cpu_dot(n: i32, x: &[T], y: &[T]) -> T;
}

pub struct Blas<T: Sized> {
    _phantom: PhantomData<T>,
}

impl<T: Sized> BlasOp<T> for Blas<T> {
    default fn caffe_cpu_dot(_n: i32, _x: &[T], _y: &[T]) -> T {
        unimplemented!();
    }
}

impl BlasOp<f32> for Blas<f32> {
    fn caffe_cpu_dot(n: i32, x: &[f32], y: &[f32]) -> f32 {
        Self::caffe_cpu_strided_dot(n, x, 1, y, 1)
    }
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

    pub fn caffe_scal(n: i32, alpha: f32, x: &mut [f32]) {
        unsafe { sscal(n, alpha, x, 1) }
    }
}

impl BlasOp<f64> for Blas<f64> {
    fn caffe_cpu_dot(n: i32, x: &[f64], y: &[f64]) -> f64 {
        Self::caffe_cpu_strided_dot(n, x, 1, y, 1)
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

    pub fn caffe_scal(n: i32, alpha: f64, x: &mut [f64]) {
        unsafe { dscal(n, alpha, x, 1) }
    }
}
