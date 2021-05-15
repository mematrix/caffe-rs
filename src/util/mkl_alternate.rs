use cblas::{sscal, dscal, saxpy, daxpy};

// Functions that caffe uses but are not present if MKL is not linked.


/// Simple macro to generate an unsafe loop on two slice with given loop size.
macro_rules! check_loop_unsafe {
    ($i:ident, $n:tt, $ex:expr) => {
        for $i in 0..$n {
            // SAFETY: the `assert` check guards that the index is not out-of-bounds.
            unsafe {
                $ex;
            }
        }
    };
    ($i:ident, $n:tt, $a:tt, $y:tt, $ex:expr) => {
        assert!($a.len() >= $n && $y.len() >= $n);
        check_loop_unsafe!($i, $n, $ex);
    };
    ($i:ident, $n:tt, $a:tt, $b:tt, $y:tt, $ex:expr) => {
        assert!($a.len() >= $n && $b.len() >= $n && $y.len() >= $n);
        check_loop_unsafe!($i, $n, $ex);
    };
}

// A simple way to define the vsl unary functions. The operation should be in the
// form e.g. y[i] = sqrt(a[i])

pub fn vs_sqr(n: usize, a: &[f32], y: &mut [f32]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = *a.get_unchecked(i) * *a.get_unchecked(i));
}

pub fn vd_sqr(n: usize, a: &[f64], y: &mut [f64]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = *a.get_unchecked(i) * *a.get_unchecked(i));
}

pub fn vs_sqrt(n: usize, a: &[f32], y: &mut [f32]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).sqrt());
}

pub fn vd_sqrt(n: usize, a: &[f64], y: &mut [f64]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).sqrt());
}

pub fn vs_exp(n: usize, a: &[f32], y: &mut [f32]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).exp());
}

pub fn vd_exp(n: usize, a: &[f64], y: &mut [f64]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).exp());
}

pub fn vs_ln(n: usize, a: &[f32], y: &mut [f32]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).ln());
}

pub fn vd_ln(n: usize, a: &[f64], y: &mut [f64]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).ln());
}

pub fn vs_abs(n: usize, a: &[f32], y: &mut [f32]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).abs());
}

pub fn vd_abs(n: usize, a: &[f64], y: &mut [f64]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).abs());
}

/// Output is 1 for the positives, 0 for zero, and -1 for the negatives.
pub fn vs_sign(n: usize, a: &[f32], y: &mut [f32]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = vs_get_sign(*a.get_unchecked(i)) as f32);
}

/// Output is 1 for the positives, 0 for zero, and -1 for the negatives.
pub fn vd_sign(n: usize, a: &[f64], y: &mut [f64]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = vd_get_sign(*a.get_unchecked(i)) as f64);
}

/// Returns 1 if the input has its sign bit set (is negative, include -0.0, NAN with neg sign).
pub fn vs_sgn_bit(n: usize, a: &[f32], y: &mut [f32]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).is_sign_negative() as i32 as f32);
}

/// Returns 1 if the input has its sign bit set (is negative, include -0.0, NAN with neg sign).
pub fn vd_sgn_bit(n: usize, a: &[f64], y: &mut [f64]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).is_sign_negative() as i32 as f64);
}

pub fn vs_fabs(n: usize, a: &[f32], y: &mut [f32]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).abs());
}

pub fn vd_fabs(n: usize, a: &[f64], y: &mut [f64]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).abs());
}

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)

pub fn vs_powx(n: usize, a: &[f32], b: f32, y: &mut [f32]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).powf(b));
}

pub fn vd_powx(n: usize, a: &[f64], b: f64, y: &mut [f64]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) = a.get_unchecked(i).powf(b));
}

// A simple way to define the vsl binary functions. The operation should be in the
// form e.g. y[i] = a[i] + b[i]

pub fn vs_add(n: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    check_loop_unsafe!(i, n, a, b, y, *y.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i));
}

pub fn vd_add(n: usize, a: &[f64], b: &[f64], y: &mut [f64]) {
    check_loop_unsafe!(i, n, a, b, y, *y.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i));
}

pub fn vs_sub(n: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    check_loop_unsafe!(i, n, a, b, y, *y.get_unchecked_mut(i) = *a.get_unchecked(i) - *b.get_unchecked(i));
}

pub fn vd_sub(n: usize, a: &[f64], b: &[f64], y: &mut [f64]) {
    check_loop_unsafe!(i, n, a, b, y, *y.get_unchecked_mut(i) = *a.get_unchecked(i) - *b.get_unchecked(i));
}

pub fn vs_mul(n: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    check_loop_unsafe!(i, n, a, b, y, *y.get_unchecked_mut(i) = *a.get_unchecked(i) * *b.get_unchecked(i));
}

pub fn vd_mul(n: usize, a: &[f64], b: &[f64], y: &mut [f64]) {
    check_loop_unsafe!(i, n, a, b, y, *y.get_unchecked_mut(i) = *a.get_unchecked(i) * *b.get_unchecked(i));
}

pub fn vs_div(n: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    check_loop_unsafe!(i, n, a, b, y, *y.get_unchecked_mut(i) = *a.get_unchecked(i) / *b.get_unchecked(i));
}

pub fn vd_div(n: usize, a: &[f64], b: &[f64], y: &mut [f64]) {
    check_loop_unsafe!(i, n, a, b, y, *y.get_unchecked_mut(i) = *a.get_unchecked(i) / *b.get_unchecked(i));
}

// AssignOps impls. The operation is in the form e.g. y[i] *= a[i]

pub fn vs_mul_assign(n: usize, y: &mut [f32], a: &[f32]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) *= *a.get_unchecked(i));
}

pub fn vd_mul_assign(n: usize, y: &mut [f64], a: &[f64]) {
    check_loop_unsafe!(i, n, a, y, *y.get_unchecked_mut(i) *= *a.get_unchecked(i));
}

// In addition, MKL comes with an additional function axpby that is not present in standard
// blas. We will simply use a two-step (inefficient, of course) way to mimic that.
pub fn cblas_saxpby(n: i32, alpha: f32, x: &[f32], inc_x: i32, beta: f32, y: &mut [f32], inc_y: i32) {
    unsafe {
        sscal(n, beta, y, inc_y);
        saxpy(n, alpha, x, inc_x, y, inc_y);
    }
}

pub fn cblas_daxpby(n: i32, alpha: f64, x: &[f64], inc_x: i32, beta: f64, y: &mut [f64], inc_y: i32) {
    unsafe {
        dscal(n, beta, y, inc_y);
        daxpy(n, alpha, x, inc_x, y, inc_y);
    }
}

// Other dependent functions.
// mark: maybe use a generic impl.
/// Output is 1 for the positives, 0 for zero, and -1 for the negatives.
pub fn vs_get_sign(val: f32) -> i8 {
    ((0f32 - val) as i8) - ((val < 0f32) as i8)
}

/// Output is 1 for the positives, 0 for zero, and -1 for the negatives.
pub fn vd_get_sign(val: f64) -> i8 {
    ((0f64 < val) as i8) - ((val < 0f64) as i8)
}
