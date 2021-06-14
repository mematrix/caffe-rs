use crate::blob::BlobType;
use crate::layer::{CaffeLayer, LayerImpl, BlobVec};
use crate::layers::neuron_layer::NeuronLayer;
use crate::proto::caffe::LayerParameter;


/// Clip: $$ y = \max(min, \min(max, x)) $$.
pub struct ClipLayer<T: BlobType> {
    layer: NeuronLayer<T>,
}

impl<T: BlobType> ClipLayer<T> {
    /// `param` provides **ClipParameter** clip_param, with **ClipLayer options**:
    /// - min
    /// - max
    pub fn new(param: &LayerParameter) -> Self {
        Self {
            layer: NeuronLayer::new(param),
        }
    }
}

impl<T: BlobType> CaffeLayer for ClipLayer<T> {
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
        "Clip"
    }

    fn exact_num_bottom_blobs(&self) -> i32 {
        self.layer.exact_num_bottom_blobs()
    }

    fn exact_num_top_blobs(&self) -> i32 {
        self.layer.exact_num_top_blobs()
    }

    fn forward_cpu(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        let b0 = bottom[0].as_ref().borrow();
        let mut t0 = top[0].borrow_mut();
        let bottom_data = b0.cpu_data();
        let count = b0.count();
        let top_data = t0.mutable_cpu_data();

        let min = self.layer.get_impl().layer_param.get_clip_param().get_min();
        let max = self.layer.get_impl().layer_param.get_clip_param().get_max();
        let min = T::from_f32(min);
        let max = T::from_f32(max);

        for i in 0..count {
            top_data[i] = T::max(min, T::min(bottom_data[i], max));
        }
    }

    fn forward_gpu(&mut self, _bottom: &BlobVec<Self::DataType>, _top: &BlobVec<Self::DataType>) {
        no_gpu!();
    }

    fn backward_cpu(&mut self, top: &BlobVec<Self::DataType>, propagate_down: &Vec<bool>,
                    bottom: &BlobVec<Self::DataType>) {
        if !propagate_down[0] {
            return;
        }

        let mut b0 = bottom[0].borrow_mut();
        let t0 = top[0].as_ref().borrow();

        let count = b0.count();
        let top_diff = t0.cpu_diff();
        let bottom_ref = b0.mutable_cpu_mem_ref();

        let min = self.layer.get_impl().layer_param.get_clip_param().get_min();
        let max = self.layer.get_impl().layer_param.get_clip_param().get_max();
        let min = T::from_f32(min);
        let max = T::from_f32(max);

        for i in 0..count {
            let data = bottom_ref.data[i];
            let in_range = (data >= min && data <= max) as i32;
            let mut value = top_diff[i];
            value *= T::from_i32(in_range);
            bottom_ref.diff[i] = value;
        }
    }

    fn backward_gpu(&mut self, _top: &BlobVec<Self::DataType>, _propagate_down: &Vec<bool>,
                    _bottom: &BlobVec<Self::DataType>) {
        no_gpu!();
    }
}

register_layer_class!(Clip);
