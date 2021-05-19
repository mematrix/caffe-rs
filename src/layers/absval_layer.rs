use super::neuron_layer::NeuronLayer;

use crate::blob::{BlobType, BlobMemRefMut};
use crate::layer::{CaffeLayer, LayerImpl, BlobVec, def_layer_setup};
use crate::proto::caffe::{LayerParameter};
use crate::util::math_functions::CaffeNum;


/// Computes $ y = |x| $.
///
/// Bottom input Blob vector (length 1)
/// - $ (N \times C \times H \times W) $ the inputs $ x $.
///
/// Top output Blob vector (length 1)
/// - $ (N \times C \times H \times W) $ the computed outputs $ y = |x| $.
#[derive(Clone)]
pub struct AbsValLayer<T: BlobType> {
    layer: NeuronLayer<T>,
}

impl<T: BlobType> AbsValLayer<T> {
    pub fn new(param: &LayerParameter) -> Self {
        AbsValLayer {
            layer: NeuronLayer::new(param)
        }
    }
}

impl<T: BlobType> CaffeLayer<T> for AbsValLayer<T> {
    fn get_impl(&self) -> &LayerImpl<T> {
        self.layer.get_impl()
    }

    fn get_impl_mut(&mut self) -> &mut LayerImpl<T> {
        self.layer.get_impl_mut()
    }

    fn layer_setup(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        def_layer_setup(self, bottom, top);
        assert_ne!(top[0].as_ptr(), bottom[0].as_ptr(),
                   "{:?} Layer does not allow in-place computation.",
                   self.layer_type());
    }

    fn reshape(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        self.layer.reshape(bottom, top);
    }

    fn layer_type(&self) -> &'static str {
        "AbsVal"
    }

    fn exact_num_bottom_blobs(&self) -> i32 {
        1
    }

    fn exact_num_top_blobs(&self) -> i32 {
        1
    }

    fn forward_cpu(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        let count = top[0].as_ref().borrow().count();
        let mut top_data = top[0].borrow_mut();
        T::caffe_abs(count, bottom[0].as_ref().borrow().cpu_data(), top_data.mutable_cpu_data());
    }

    fn forward_gpu(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        no_gpu!();
    }

    fn backward_cpu(&mut self, top: &BlobVec<T>, propagate_down: &Vec<bool>, bottom: &BlobVec<T>) {
        if propagate_down[0] {
            let count = top[0].as_ref().borrow().count();
            let top_diff = top[0].as_ref().borrow();
            let top_diff = top_diff.cpu_diff();
            let mut bottom_mut = bottom[0].borrow_mut();
            let bottom_mut = bottom_mut.mutable_cpu_mem_ref();
            T::caffe_cpu_sign(count, bottom_mut.data, bottom_mut.diff);
            T::caffe_mul_assign(count, bottom_mut.diff, top_diff);
        }
    }

    fn backward_gpu(&mut self, top: &BlobVec<T>, propagate_down: &Vec<bool>, bottom: &BlobVec<T>) {
        no_gpu!();
    }
}

register_layer_class!(AbsVal);
