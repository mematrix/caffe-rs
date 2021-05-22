use std::cell::RefCell;
use std::rc::Rc;

use rand::RngCore;
use rand::distributions::{Uniform, Distribution};

use crate::common::{CaffeRng, Caffe};


pub fn caffe_rng() -> Rc<RefCell<CaffeRng>> {
    Caffe::rng()
}

pub fn shuffle<T>(slice: &mut [T], gen: &mut dyn RngCore) {
    if slice.len() < 1 {
        return;
    }

    for i in (1..slice.len()).rev() {
        let dist = Uniform::new(0, i);
        slice.swap(i, dist.sample(gen));
    }
}

pub fn shuffle_caffe_rng<T>(slice: &mut [T]) {
    shuffle(slice, caffe_rng().as_ref().borrow_mut().generator());
}
