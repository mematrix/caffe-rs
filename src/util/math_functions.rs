use std::ops::{AddAssign, Div};

use cblas::{saxpy, daxpy, sasum, dasum, sdot, ddot, sscal, dscal};

use super::mkl_alternate::*;


pub trait CaffeNum: Copy + Sized + AddAssign + Div {
    fn is_zero(&self) -> bool;

    fn from_f64(v: f64) -> Self;

    fn from_f32(v: f32) -> Self;

    fn from_usize(v: usize) -> Self;

    fn to_f64(self) -> f64;

    fn to_f32(self) -> f32;

    fn from_div(v: <Self as Div<Self>>::Output) -> Self;

    fn sqrt(v: Self) -> Self;

    // define the functions associated with Self

    fn caffe_axpy(n: i32, alpha: Self, x: &[Self], y: &mut [Self]);

    fn caffe_cpu_asum(n: i32, x: &[Self]) -> Self;

    fn caffe_cpu_strided_dot(n: i32, x: &[Self], inc_x: i32, y: &[Self], inc_y: i32) -> Self;

    fn caffe_scal(n: i32, alpha: Self, x: &mut [Self]);

    fn caffe_cpu_dot(n: i32, x: &[Self], y: &[Self]) -> Self;

    fn caffe_add(n: usize, a: &[Self], b: &[Self], y: &mut [Self]);

    fn caffe_sub(n: usize, a: &[Self], b: &[Self], y: &mut [Self]);

    fn caffe_mul(n: usize, a: &[Self], b: &[Self], y: &mut [Self]);

    fn caffe_mul_assign(n: usize, y: &mut [Self], a: &[Self]);

    fn caffe_div(n: usize, a: &[Self], b: &[Self], y: &mut [Self]);

    fn caffe_powx(n: usize, a: &[Self], b: Self, y: &mut [Self]);

    fn caffe_sqr(n: usize, a: &[Self], y: &mut [Self]);

    fn caffe_sqrt(n: usize, a: &[Self], y: &mut [Self]);

    fn caffe_exp(n: usize, a: &[Self], y: &mut [Self]);

    fn caffe_log(n: usize, a: &[Self], y: &mut [Self]);

    fn caffe_abs(n: usize, a: &[Self], y: &mut [Self]);

    fn caffe_cpu_sign(n: usize, x: &[Self], y: &mut [Self]);

    fn caffe_cpu_sgnbit(n: usize, x: &[Self], y: &mut [Self]);

    fn caffe_cpu_fabs(n: usize, x: &[Self], y: &mut [Self]);
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

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_div(v: Self::Output) -> Self {
        v
    }

    fn sqrt(v: Self) -> Self {
        (v as f64).sqrt() as i32
    }

    fn caffe_axpy(n: i32, alpha: Self, x: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_cpu_asum(n: i32, x: &[Self]) -> Self {
        todo!()
    }

    fn caffe_cpu_strided_dot(n: i32, x: &[Self], inc_x: i32, y: &[Self], inc_y: i32) -> Self {
        todo!()
    }

    fn caffe_scal(n: i32, alpha: Self, x: &mut [Self]) {
        todo!()
    }

    fn caffe_cpu_dot(n: i32, x: &[Self], y: &[Self]) -> Self {
        todo!()
    }

    fn caffe_add(n: usize, a: &[Self], b: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_sub(n: usize, a: &[Self], b: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_mul(n: usize, a: &[Self], b: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_mul_assign(n: usize, y: &mut [Self], a: &[Self]) {
        todo!()
    }

    fn caffe_div(n: usize, a: &[Self], b: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_powx(n: usize, a: &[Self], b: Self, y: &mut [Self]) {
        todo!()
    }

    fn caffe_sqr(n: usize, a: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_sqrt(n: usize, a: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_exp(n: usize, a: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_log(n: usize, a: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_abs(n: usize, a: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_cpu_sign(n: usize, x: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_cpu_sgnbit(n: usize, x: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_cpu_fabs(n: usize, x: &[Self], y: &mut [Self]) {
        todo!()
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

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn to_f32(self) -> f32 {
        self
    }

    fn from_div(v: Self::Output) -> Self {
        v
    }

    fn sqrt(v: Self) -> Self {
        v.sqrt()
    }

    // Impls functions for Self

    fn caffe_axpy(n: i32, alpha: f32, x: &[f32], y: &mut [f32]) {
        unsafe { saxpy(n, alpha, x, 1, y, 1); }
    }

    fn caffe_cpu_asum(n: i32, x: &[f32]) -> f32 {
        unsafe { sasum(n, x, 1) }
    }

    fn caffe_cpu_strided_dot(n: i32, x: &[f32], inc_x: i32, y: &[f32], inc_y: i32) -> f32 {
        unsafe { sdot(n, x, inc_x, y, inc_y) }
    }

    fn caffe_scal(n: i32, alpha: f32, x: &mut [f32]) {
        unsafe { sscal(n, alpha, x, 1) }
    }

    fn caffe_cpu_dot(n: i32, x: &[f32], y: &[f32]) -> f32 {
        Self::caffe_cpu_strided_dot(n, x, 1, y, 1)
    }

    fn caffe_add(n: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
        vs_add(n, a, b, y);
    }

    fn caffe_sub(n: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
        vs_sub(n, a, b, y);
    }

    fn caffe_mul(n: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
        vs_mul(n, a, b, y);
    }

    fn caffe_mul_assign(n: usize, y: &mut [f32], a: &[f32]) {
        vs_mul_assign(n, y, a);
    }

    fn caffe_div(n: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
        vs_div(n, a, b, y);
    }

    fn caffe_powx(n: usize, a: &[f32], b: f32, y: &mut [f32]) {
        vs_powx(n, a, b, y);
    }

    fn caffe_sqr(n: usize, a: &[f32], y: &mut [f32]) {
        vs_sqr(n, a, y);
    }

    fn caffe_sqrt(n: usize, a: &[f32], y: &mut [f32]) {
        vs_sqrt(n, a, y);
    }

    fn caffe_exp(n: usize, a: &[f32], y: &mut [f32]) {
        vs_exp(n, a, y);
    }

    fn caffe_log(n: usize, a: &[f32], y: &mut [f32]) {
        vs_ln(n, a, y);
    }

    fn caffe_abs(n: usize, a: &[f32], y: &mut [f32]) {
        vs_abs(n, a, y);
    }

    fn caffe_cpu_sign(n: usize, x: &[f32], y: &mut [f32]) {
        vs_sign(n, x, y);
    }

    fn caffe_cpu_sgnbit(n: usize, x: &[f32], y: &mut [f32]) {
        vs_sgn_bit(n, x, y);
    }

    fn caffe_cpu_fabs(n: usize, x: &[f32], y: &mut [f32]) {
        vs_fabs(n, x, y);
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

    fn to_f64(self) -> f64 {
        self
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_div(v: Self::Output) -> Self {
        v
    }

    fn sqrt(v: Self) -> Self {
        v.sqrt()
    }

    // Impls Self functions.

    fn caffe_axpy(n: i32, alpha: f64, x: &[f64], y: &mut [f64]) {
        unsafe { daxpy(n, alpha, x, 1, y, 1); }
    }

    fn caffe_cpu_asum(n: i32, x: &[f64]) -> f64 {
        unsafe { dasum(n, x, 1) }
    }

    fn caffe_cpu_strided_dot(n: i32, x: &[f64], inc_x: i32, y: &[f64], inc_y: i32) -> f64 {
        unsafe { ddot(n, x, inc_x, y, inc_y) }
    }

    fn caffe_scal(n: i32, alpha: f64, x: &mut [f64]) {
        unsafe { dscal(n, alpha, x, 1) }
    }

    fn caffe_cpu_dot(n: i32, x: &[f64], y: &[f64]) -> f64 {
        Self::caffe_cpu_strided_dot(n, x, 1, y, 1)
    }

    fn caffe_add(n: usize, a: &[f64], b: &[f64], y: &mut [f64]) {
        vd_add(n, a, b, y);
    }

    fn caffe_sub(n: usize, a: &[f64], b: &[f64], y: &mut [f64]) {
        vd_sub(n, a, b, y);
    }

    fn caffe_mul(n: usize, a: &[f64], b: &[f64], y: &mut [f64]) {
        vd_mul(n, a, b, y);
    }

    fn caffe_mul_assign(n: usize, y: &mut [f64], a: &[f64]) {
        vd_mul_assign(n, y, a);
    }

    fn caffe_div(n: usize, a: &[f64], b: &[f64], y: &mut [f64]) {
        vd_div(n, a, b, y);
    }

    fn caffe_powx(n: usize, a: &[f64], b: f64, y: &mut [f64]) {
        vd_powx(n, a, b, y);
    }

    fn caffe_sqr(n: usize, a: &[f64], y: &mut [f64]) {
        vd_sqr(n, a, y);
    }

    fn caffe_sqrt(n: usize, a: &[f64], y: &mut [f64]) {
        vd_sqrt(n, a, y);
    }

    fn caffe_exp(n: usize, a: &[f64], y: &mut [f64]) {
        vd_exp(n, a, y);
    }

    fn caffe_log(n: usize, a: &[f64], y: &mut [f64]) {
        vd_ln(n, a, y);
    }

    fn caffe_abs(n: usize, a: &[f64], y: &mut [f64]) {
        vd_abs(n, a, y);
    }

    fn caffe_cpu_sign(n: usize, x: &[f64], y: &mut [f64]) {
        vd_sign(n, x, y);
    }

    fn caffe_cpu_sgnbit(n: usize, x: &[f64], y: &mut [f64]) {
        vd_sgn_bit(n, x, y);
    }

    fn caffe_cpu_fabs(n: usize, x: &[f64], y: &mut [f64]) {
        vd_fabs(n, x, y);
    }
}


pub fn caffe_copy<T: CaffeNum>(n: usize, x: &[T], y: &mut [T]) {
    if x.as_ptr() != y.as_ptr() {
        debug_assert!(x.len() >= n && y.len() >= n);

        y[..n].copy_from_slice(&x[..n]);
    }
}

pub fn caffe_set<T: CaffeNum>(n: usize, alpha: T, y: &mut [T]) {
    assert!(y.len() >= n);
    if alpha.is_zero() {
        unsafe { std::ptr::write_bytes(y.as_mut_ptr(), 0u8, n); }
    } else {
        for x in &mut y[0..n] {
            *x = alpha;
        }
    }
}
