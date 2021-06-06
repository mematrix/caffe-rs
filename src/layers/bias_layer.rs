use crate::blob::{BlobType, Blob};
use crate::layer::{LayerImpl, CaffeLayer, BlobVec, def_layer_setup, make_shared_blob};
use crate::proto::caffe::LayerParameter;
use crate::filler::get_filler;


pub struct BiasLayer<T: BlobType> {
    layer: LayerImpl<T>,
    bias_multiplier: Blob<T>,
    outer_dim: i32,
    bias_dim: i32,
    inner_dim: i32,
    dim: i32,
}

impl<T: BlobType> BiasLayer<T> {
    pub fn new(param: &LayerParameter) -> Self {
        Self {
            layer: LayerImpl::new(param),
            bias_multiplier: Blob::new(),
            outer_dim: 0,
            bias_dim: 0,
            inner_dim: 0,
            dim: 0,
        }
    }
}

impl<T: BlobType> CaffeLayer for BiasLayer<T> {
    type DataType = T;

    fn get_impl(&self) -> &LayerImpl<Self::DataType> {
        &self.layer
    }

    fn get_impl_mut(&mut self) -> &mut LayerImpl<Self::DataType> {
        & mut self.layer
    }

    fn layer_setup(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        if bottom.len() == 1 && !self.layer.blobs.is_empty() {
            info!("Skipping parameter initialization");
        } else if bottom.len() == 1 {
            // bias is a learned parameter; initialize it
            let param = self.layer.layer_param.get_bias_param();
            let b0 = bottom[0].as_ref().borrow();
            let axis = b0.canonical_axis_index(param.get_axis());
            let num_axes = param.get_num_axes();
            assert!(num_axes >= -1, "num_axes must be non-negative, or -1 to extend to the end of bottom[0]");
            if num_axes >= 0 {
                assert!(b0.num_axes() >= (axis as i32 + num_axes),
                        "bias blob's shape extends past bottom[0]'s shape when applied \
                        starting with bottom[0] axis = {}",
                        axis);
            }

            let bias_shape = if num_axes == -1 {
                &b0.shape()[axis..]
            } else {
                &b0.shape()[axis..(axis + num_axes as usize)]
            };
            let mut blob = Blob::with_shape(bias_shape);
            let filler = get_filler(param.get_filler());
            filler.fill(&mut blob);
            self.layer.blobs.push(make_shared_blob(blob));
        }

        self.layer.param_propagate_down.resize(self.layer.blobs.len(), true);
    }

    fn reshape(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        todo!()
    }

    fn layer_type(&self) -> &'static str {
        "Bias"
    }

    fn min_bottom_blobs(&self) -> i32 {
        1
    }

    fn max_bottom_blobs(&self) -> i32 {
        2
    }

    fn exact_num_top_blobs(&self) -> i32 {
        1
    }

    fn forward_cpu(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        todo!()
    }

    fn forward_gpu(&mut self, _bottom: &BlobVec<Self::DataType>, _top: &BlobVec<Self::DataType>) {
        no_gpu!();
    }

    fn backward_cpu(&mut self, top: &BlobVec<Self::DataType>, propagate_down: &Vec<bool>, bottom: &BlobVec<Self::DataType>) {
        todo!()
    }

    fn backward_gpu(&mut self, _top: &BlobVec<Self::DataType>, _propagate_down: &Vec<bool>, _bottom: &BlobVec<Self::DataType>) {
        no_gpu!();
    }
}
