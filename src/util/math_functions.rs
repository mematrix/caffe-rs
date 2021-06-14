use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign, Neg, Add, Sub, Mul, Div};

use cblas::{Transpose, Layout, saxpy, daxpy, sasum, dasum, sdot, ddot, sscal, dscal, sgemm, sgemv, dgemm, dgemv, scopy, dcopy};
use float_next_after::NextAfter;
use rand::distributions::{Uniform, Distribution, Bernoulli};
use rand::distributions::uniform::SampleUniform;
use rand_distr::Normal;

use super::mkl_alternate::*;
use crate::util::rng::caffe_rng;


pub trait CaffeNum:
    Copy + Sized + Default + PartialOrd +
    AddAssign + SubAssign + MulAssign + DivAssign + Neg<Output = Self> +
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> +
    SampleUniform {
    fn is_zero(&self) -> bool;

    fn is_nan_v(&self) -> bool;

    fn from_f64(v: f64) -> Self;

    fn from_f32(v: f32) -> Self;

    fn from_i32(v: i32) -> Self;

    fn from_usize(v: usize) -> Self;

    fn to_f64(self) -> f64;

    fn to_f32(self) -> f32;

    fn to_i32(self) -> i32;

    fn to_usize(self) -> usize;

    fn sqrt(v: Self) -> Self;

    /// Return nature logarithm of the number.
    fn ln(v: Self) -> Self;

    /// Return `e^v`.
    fn exp(v: Self) -> Self;

    fn fabs(v: Self) -> Self;

    fn min(a: Self, b: Self) -> Self {
        if b < a { b } else { a }
    }

    fn max(a: Self, b: Self) -> Self {
        if a < b { b } else { a }
    }

    /// Function likes the C++ `std::nextafter`, provided the next representable value of `self`
    /// toward the `y` direction.
    fn next_toward(self, y: Self) -> Self;

    /// Function likes the C++ `std::numeric_limits<Self>::max()`.
    /// Returns the max value of Self Type.
    fn num_max() -> Self;

    // define the functions associated with Self

    fn caffe_axpy(n: i32, alpha: Self, x: &[Self], y: &mut [Self]);

    fn caffe_cpu_axpby(n: i32, alpha: Self, x: &[Self], beta: Self, y: &mut [Self]);

    fn caffe_cpu_asum(n: i32, x: &[Self]) -> Self;

    fn caffe_cpu_strided_dot(n: i32, x: &[Self], inc_x: i32, y: &[Self], inc_y: i32) -> Self;

    fn caffe_scal(n: i32, alpha: Self, x: &mut [Self]);

    fn caffe_cpu_dot(n: i32, x: &[Self], y: &[Self]) -> Self;

    fn caffe_cpu_scale(n: i32, alpha: Self, x: &[Self], y: &mut [Self]);

    fn caffe_add(n: usize, a: &[Self], b: &[Self], y: &mut [Self]);

    fn caffe_sub(n: usize, a: &[Self], b: &[Self], y: &mut [Self]);

    fn caffe_sub_assign(n: usize, y: &mut [Self], a: &[Self]);

    fn caffe_mul(n: usize, a: &[Self], b: &[Self], y: &mut [Self]);

    fn caffe_mul_assign(n: usize, y: &mut [Self], a: &[Self]);

    fn caffe_div(n: usize, a: &[Self], b: &[Self], y: &mut [Self]);

    fn caffe_add_scalar(n: usize, alpha: Self, y: &mut [Self]);

    fn caffe_powx(n: usize, a: &[Self], b: Self, y: &mut [Self]);

    fn caffe_sqr(n: usize, a: &[Self], y: &mut [Self]);

    fn caffe_sqrt(n: usize, a: &[Self], y: &mut [Self]);

    fn caffe_exp(n: usize, a: &[Self], y: &mut [Self]);

    fn caffe_log(n: usize, a: &[Self], y: &mut [Self]);

    fn caffe_abs(n: usize, a: &[Self], y: &mut [Self]);

    fn caffe_cpu_sign(n: usize, x: &[Self], y: &mut [Self]);

    fn caffe_cpu_sgnbit(n: usize, x: &[Self], y: &mut [Self]);

    fn caffe_cpu_fabs(n: usize, x: &[Self], y: &mut [Self]);

    fn caffe_cpu_gemm(trans_a: Transpose, trans_b: Transpose, m: i32, n: i32, k: i32,
                      alpha: Self, a: &[Self], b: &[Self], beta: Self, c: &mut [Self]);

    fn caffe_cpu_gemv(trans_a: Transpose, m: i32, n: i32, alpha: Self,
                      a: &[Self], x: &[Self], beta: Self, y: &mut [Self]);
}

impl CaffeNum for i32 {
    fn is_zero(&self) -> bool {
        *self == 0
    }

    fn is_nan_v(&self) -> bool {
        false
    }

    fn from_f64(v: f64) -> Self {
        v as i32
    }

    fn from_f32(v: f32) -> Self {
        v as i32
    }

    fn from_i32(v: i32) -> Self {
        v
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

    fn to_i32(self) -> i32 {
        self
    }

    fn to_usize(self) -> usize {
        self as usize
    }

    fn sqrt(v: Self) -> Self {
        (v as f64).sqrt() as i32
    }

    fn ln(v: Self) -> Self {
        (v as f64).ln() as i32
    }

    fn exp(v: Self) -> Self {
        (v as f64).exp() as i32
    }

    fn fabs(v: Self) -> Self {
        v.abs()
    }

    fn next_toward(self, y: Self) -> Self {
        if self == y {
            return self;
        }
        if self > y {
            self - 1
        } else {
            self + 1
        }
    }

    fn num_max() -> Self {
        i32::MAX
    }

    fn caffe_axpy(n: i32, alpha: Self, x: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_cpu_axpby(n: i32, alpha: Self, x: &[Self], beta: Self, y: &mut [Self]) {
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

    fn caffe_cpu_scale(n: i32, alpha: Self, x: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_add(n: usize, a: &[Self], b: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_sub(n: usize, a: &[Self], b: &[Self], y: &mut [Self]) {
        todo!()
    }

    fn caffe_sub_assign(n: usize, y: &mut [Self], a: &[Self]) {
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

    fn caffe_add_scalar(n: usize, alpha: Self, y: &mut [Self]) {
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

    fn caffe_cpu_gemm(trans_a: Transpose, trans_b: Transpose, m: i32, n: i32, k: i32,
                      alpha: Self, a: &[Self], b: &[Self], beta: Self, c: &mut [Self]) {
        todo!()
    }

    fn caffe_cpu_gemv(trans_a: Transpose, m: i32, n: i32, alpha: Self,
                      a: &[Self], x: &[Self], beta: Self, y: &mut [Self]) {
        todo!()
    }
}

impl CaffeNum for f32 {
    fn is_zero(&self) -> bool {
        *self == 0f32
    }

    fn is_nan_v(&self) -> bool {
        self.is_nan()
    }

    fn from_f64(v: f64) -> Self {
        v as f32
    }

    fn from_f32(v: f32) -> Self {
        v
    }

    fn from_i32(v: i32) -> Self {
        v as f32
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

    fn to_i32(self) -> i32 {
        self as i32
    }

    fn to_usize(self) -> usize {
        self as usize
    }

    fn sqrt(v: Self) -> Self {
        v.sqrt()
    }

    fn ln(v: Self) -> Self {
        v.ln()
    }

    fn exp(v: Self) -> Self {
        v.exp()
    }

    fn fabs(v: Self) -> Self {
        v.abs()
    }

    fn next_toward(self, y: Self) -> Self {
        self.next_after(y)
    }

    fn num_max() -> Self {
        f32::MAX
    }

    // Impls functions for Self

    fn caffe_axpy(n: i32, alpha: f32, x: &[f32], y: &mut [f32]) {
        unsafe { saxpy(n, alpha, x, 1, y, 1); }
    }

    fn caffe_cpu_axpby(n: i32, alpha: Self, x: &[Self], beta: Self, y: &mut [Self]) {
        cblas_saxpby(n, alpha, x, 1, beta, y, 1);
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

    fn caffe_cpu_scale(n: i32, alpha: Self, x: &[Self], y: &mut [Self]) {
        unsafe {
            scopy(n, x, 1, y, 1);
            sscal(n, alpha, y, 1);
        }
    }

    fn caffe_add(n: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
        vs_add(n, a, b, y);
    }

    fn caffe_sub(n: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
        vs_sub(n, a, b, y);
    }

    fn caffe_sub_assign(n: usize, y: &mut [Self], a: &[Self]) {
        vs_sub_assign(n, y, a);
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

    fn caffe_add_scalar(n: usize, alpha: Self, y: &mut [Self]) {
        assert!(y.len() >= n);
        for i in 0..n {
            // SAFETY: assert y.len >= n
            unsafe { *y.get_unchecked_mut(i) += alpha; }
        }
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

    fn caffe_cpu_gemm(trans_a: Transpose, trans_b: Transpose, m: i32, n: i32, k: i32,
                      alpha: Self, a: &[Self], b: &[Self], beta: Self, c: &mut [Self]) {
        let lda = if trans_a == Transpose::None { k } else { m };
        let ldb = if trans_b == Transpose::None { n } else { k };
        unsafe { sgemm(Layout::RowMajor, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, n); }
    }

    fn caffe_cpu_gemv(trans_a: Transpose, m: i32, n: i32, alpha: Self,
                      a: &[Self], x: &[Self], beta: Self, y: &mut [Self]) {
        unsafe { sgemv(Layout::RowMajor, trans_a, m, n, alpha, a, n, x, 1, beta, y, 1); }
    }
}

impl CaffeNum for f64 {
    fn is_zero(&self) -> bool {
        *self == 0f64
    }

    fn is_nan_v(&self) -> bool {
        self.is_nan()
    }

    fn from_f64(v: f64) -> Self {
        v
    }

    fn from_f32(v: f32) -> Self {
        v as f64
    }

    fn from_i32(v: i32) -> Self {
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

    fn to_i32(self) -> i32 {
        self as i32
    }

    fn to_usize(self) -> usize {
        self as usize
    }

    fn sqrt(v: Self) -> Self {
        v.sqrt()
    }

    fn ln(v: Self) -> Self {
        v.ln()
    }

    fn exp(v: Self) -> Self {
        v.exp()
    }

    fn fabs(v: Self) -> Self {
        v.abs()
    }

    fn next_toward(self, y: Self) -> Self {
        self.next_after(y)
    }

    fn num_max() -> Self {
        f64::MAX
    }

    // Impls Self functions.

    fn caffe_axpy(n: i32, alpha: f64, x: &[f64], y: &mut [f64]) {
        unsafe { daxpy(n, alpha, x, 1, y, 1); }
    }

    fn caffe_cpu_axpby(n: i32, alpha: Self, x: &[Self], beta: Self, y: &mut [Self]) {
        cblas_daxpby(n, alpha, x, 1, beta, y, 1);
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

    fn caffe_cpu_scale(n: i32, alpha: Self, x: &[Self], y: &mut [Self]) {
        unsafe {
            dcopy(n, x, 1, y, 1);
            dscal(n, alpha, y, 1);
        }
    }

    fn caffe_add(n: usize, a: &[f64], b: &[f64], y: &mut [f64]) {
        vd_add(n, a, b, y);
    }

    fn caffe_sub(n: usize, a: &[f64], b: &[f64], y: &mut [f64]) {
        vd_sub(n, a, b, y);
    }

    fn caffe_sub_assign(n: usize, y: &mut [Self], a: &[Self]) {
        vd_sub_assign(n, y, a);
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

    fn caffe_add_scalar(n: usize, alpha: f64, y: &mut [f64]) {
        assert!(y.len() >= n);
        for i in 0..n {
            // SAFETY: assert y.len >= n
            unsafe { *y.get_unchecked_mut(i) += alpha; }
        }
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

    fn caffe_cpu_gemm(trans_a: Transpose, trans_b: Transpose, m: i32, n: i32, k: i32,
                      alpha: Self, a: &[Self], b: &[Self], beta: Self, c: &mut [Self]) {
        let lda = if trans_a == Transpose::None { k } else { m };
        let ldb = if trans_b == Transpose::None { n } else { k };
        unsafe { dgemm(Layout::RowMajor, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, n); }
    }

    fn caffe_cpu_gemv(trans_a: Transpose, m: i32, n: i32, alpha: Self,
                      a: &[Self], x: &[Self], beta: Self, y: &mut [Self]) {
        unsafe { dgemv(Layout::RowMajor, trans_a, m, n, alpha, a, n, x, 1, beta, y, 1); }
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
        // SAFETY: the assert check makes sure that slice memory size is valid and `y`
        // is allocated in the safety context that guards the memory alignment.
        unsafe { std::ptr::write_bytes(y.as_mut_ptr(), 0u8, n); }
    } else {
        for x in &mut y[0..n] {
            *x = alpha;
        }
    }
}

pub fn caffe_next_after<T: CaffeNum>(b: T) -> T {
    b.next_toward(T::num_max())
}

pub fn caffe_rng_uniform<T: CaffeNum>(n: usize, a: T, b: T, r: &mut [T]) {
    assert!(a <= b);
    assert!(n <= r.len());
    let random_distribution = Uniform::new(a, caffe_next_after(b));
    let rng = caffe_rng();
    let mut rng = rng.borrow_mut();
    for i in 0..n {
        // SAFETY: the assert check makes sure that index `i` is between in the slice range.
        unsafe { *r.get_unchecked_mut(i) = random_distribution.sample(rng.generator()) }
    }
}

pub fn caffe_rng_gaussian<T: CaffeNum>(n: usize, a: T, sigma: T, r: &mut [T]) {
    let a = a.to_f64();
    let sigma = sigma.to_f64();
    assert!(sigma > 0f64);
    assert!(n <= r.len());
    let random_distribution = Normal::new(a, sigma).unwrap();
    let rng = caffe_rng();
    let mut rng = rng.borrow_mut();
    for i in 0..n {
        // SAFETY: the assert check makes sure that index `i` is between in the slice range.
        unsafe { *r.get_unchecked_mut(i) = T::from_f64(random_distribution.sample(rng.generator())); }
    }
}

pub fn caffe_rng_bernoulli_i32<T: CaffeNum>(n: usize, p: T, r: &mut [i32]) {
    let p = p.to_f64();
    assert!(p >= 0f64 && p <= 1f64);
    let random_distribution = Bernoulli::new(p).unwrap();
    let rng = caffe_rng();
    let mut rng = rng.borrow_mut();
    for i in 0..n {
        // SAFETY: the assert check makes sure that index `i` is between in the slice range.
        unsafe { *r.get_unchecked_mut(i) = random_distribution.sample(rng.generator()) as i32; }
    }
}

pub fn caffe_rng_bernoulli_u32<T: CaffeNum>(n: usize, p: T, r: &mut [u32]) {
    let p = p.to_f64();
    assert!(p >= 0f64 && p <= 1f64);
    let random_distribution = Bernoulli::new(p).unwrap();
    let rng = caffe_rng();
    let mut rng = rng.borrow_mut();
    for i in 0..n {
        // SAFETY: the assert check makes sure that index `i` is between in the slice range.
        unsafe { *r.get_unchecked_mut(i) = random_distribution.sample(rng.generator()) as u32; }
    }
}
