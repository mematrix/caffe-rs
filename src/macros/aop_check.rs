
macro_rules! impl_op_check {
    ($op:tt, $left:expr, $right:expr) => {
        if !($left $op $right) {
            let lv = $left;
            let rv = $right;
            panic!("check failed: {:?}({:?}) {:?} {:?}({:?}).",
                   stringify!($left),
                   lv,
                   stringify!($op),
                   stringify!($right),
                   rv);
        }
    };
    ($op:tt, $left:expr, $right:expr, $msg:literal) => {
        if !($left $op $right) {
            let lv = $left;
            let rv = $right;
            panic!("check failed: {:?}({:?}) {:?} {:?}({:?}). msg: {:?}",
                   stringify!($left),
                   lv,
                   stringify!($op),
                   stringify!($right),
                   rv,
                   $msg);
        }
    };
    ($op:tt, $left:expr, $right:expr, $fmt:literal, $($element:expr),*) => {
        if !($left $op $right) {
            let lv = $left;
            let rv = $right;
            panic!("check failed: {:?}({:?}) {:?} {:?}({:?}). msg: {:?}",
                   stringify!($left),
                   lv,
                   stringify!($op),
                   stringify!($right),
                   rv,
                   format!($fmt, $($element),*));
        }
    }
}

// #[macro_export]
macro_rules! check_eq {
    ($left:expr, $right:expr, $($e:expr),*) => {
        impl_op_check!(==, $left, $right, $($e),*);
    };
    ($left:expr, $right:expr) => {
        impl_op_check!(==, $left, $right);
    };
}

// #[macro_export]
macro_rules! check_le {
    ($left:expr, $right:expr, $($e:expr),*) => {
        impl_op_check!(<=, $left, $right, $($e),*);
    };
    ($left:expr, $right:expr) => {
        impl_op_check!(<=, $left, $right);
    };
}

macro_rules! check_lt {
    ($left:expr, $right:expr, $($e:expr),*) => {
        impl_op_check!(<, $left, $right, $($e),*);
    };
    ($left:expr, $right:expr) => {
        impl_op_check!(<, $left, $right);
    };
}

macro_rules! check_ge {
    ($left:expr, $right:expr, $($e:expr),*) => {
        impl_op_check!(>=, $left, $right, $($e),*);
    };
    ($left:expr, $right:expr) => {
        impl_op_check!(>=, $left, $right);
    };
}

macro_rules! check_gt {
    ($left:expr, $right:expr, $($e:expr),*) => {
        impl_op_check!(>, $left, $right, $($e),*);
    };
    ($left:expr, $right:expr) => {
        impl_op_check!(>, $left, $right);
    };
}
