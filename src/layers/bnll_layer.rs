use crate::blob::BlobType;
use crate::layer::{CaffeLayer, LayerImpl, BlobVec};
use crate::layers::neuron_layer::NeuronLayer;
use crate::proto::caffe::LayerParameter;


pub struct BNLLLayer<T: BlobType> {
    layer: NeuronLayer<T>,
}

impl<T: BlobType> BNLLLayer<T> {
    pub fn new(param: &LayerParameter) -> Self {
        Self {
            layer: NeuronLayer::new(param),
        }
    }
}

const BNLL_THRESHOLD: f64 = 50.0;

impl<T: BlobType> CaffeLayer for BNLLLayer<T> {
    type DataType = T;

    fn get_impl(&self) -> &LayerImpl<Self::DataType> {
        self.layer.get_impl()
    }

    fn get_impl_mut(&mut self) -> &mut LayerImpl<Self::DataType> {
        self.layer.get_impl_mut()
    }

    fn reshape(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        self.layer.reshape(bottom, top);
    }

    fn layer_type(&self) -> &'static str {
        "BNLL"
    }

    fn exact_num_bottom_blobs(&self) -> i32 {
        self.layer.exact_num_bottom_blobs()
    }

    fn exact_num_top_blobs(&self) -> i32 {
        self.layer.exact_num_bottom_blobs()
    }

    fn forward_cpu(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        let b0 = bottom[0].as_ref().borrow();
        let mut t0 = top[0].borrow_mut();
        let count = b0.count();
        let bottom_data = b0.cpu_data();
        let top_data = t0.mutable_cpu_data();
        for i in 0..count {
            let bottom = bottom_data[i];
            top_data[i] = if bottom > T::default() {
                bottom + T::ln(T::exp(-bottom) + T::from_i32(1))
            } else {
                T::ln(T::exp(bottom) + T::from_i32(1))
            };
        }
    }

    fn forward_gpu(&mut self, _bottom: &BlobVec<Self::DataType>, _top: &BlobVec<Self::DataType>) {
        no_gpu!();
    }

    fn backward_cpu(&mut self, top: &BlobVec<Self::DataType>, propagate_down: &Vec<bool>, bottom: &BlobVec<Self::DataType>) {
        if !propagate_down[0] {
            return;
        }

        let t0 = top[0].as_ref().borrow();
        let mut b0 = bottom[0].borrow_mut();
        let top_diff = t0.cpu_diff();
        let count = b0.count();
        let mem_ref = b0.mutable_cpu_mem_ref();
        let threshold = T::from_f64(BNLL_THRESHOLD);
        for i in 0..count {
            let exp_val = T::exp(T::min(mem_ref.data[i], threshold));
            mem_ref.diff[i] = (top_diff[i] * exp_val) / (exp_val + T::from_i32(1));
        }
    }

    fn backward_gpu(&mut self, _top: &BlobVec<Self::DataType>, _propagate_down: &Vec<bool>, _bottom: &BlobVec<Self::DataType>) {
        no_gpu!();
    }
}

register_layer_class!(BNLL);
