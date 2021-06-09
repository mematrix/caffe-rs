use std::rc::Rc;

use cblas::Transpose;

use crate::blob::{BlobType, Blob};
use crate::filler::get_filler;
use crate::layer::{LayerImpl, CaffeLayer, BlobVec, def_layer_setup, make_shared_blob};
use crate::proto::caffe::LayerParameter;
use crate::util::math_functions::{caffe_set, caffe_copy};


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
        let param = self.layer.layer_param.get_bias_param();
        let b1 = bottom.get(1).map(|b| b.as_ref().borrow());
        let blobs = &self.layer.blobs;
        let bias = b1.unwrap_or_else(|| blobs[0].as_ref().borrow());
        // Always set axis == 0 in special case where bias is a scalar
        // (num_axes == 0). Mathematically equivalent for any choice of axis, so the
        // actual setting can be safely ignored; and computation is most efficient
        // with axis == 0 and (therefore) outer_dim_ == 1.
        let b0 = bottom[0].as_ref().borrow();
        let axis = if bias.num_axes() == 0 {
            0
        } else {
            b0.canonical_axis_index(param.get_axis()) as i32
        };
        assert!(b0.num_axes() >= axis + bias.num_axes(),
                "bias blob's shape extends past bottom[0]'s shape when applied starting with bottom[0] axis = {}",
                axis);
        for i in 0..bias.num_axes() {
            assert_eq!(b0.shape_idx(axis + i), bias.shape_idx(i),
                       "dimension mismatch between bottom[0]->shape({}) and bias->shape({})",
                       axis + i, i);
        }

        self.outer_dim = b0.count_range(0, axis as usize);
        self.bias_dim = bias.count() as i32;
        self.inner_dim = b0.count_range_to_end((axis + bias.num_axes()) as usize);
        self.dim = self.bias_dim * self.inner_dim;
        if !Rc::ptr_eq(&bottom[0], &top[0]) {
            top[0].borrow_mut().reshape_like(&*bottom[0].as_ref().borrow());
        }
        self.bias_multiplier.reshape(&[self.inner_dim]);
        if self.bias_multiplier.cpu_data()[self.inner_dim as usize - 1usize] != T::from_i32(1) {
            caffe_set(self.inner_dim as usize, T::from_i32(1), self.bias_multiplier.mutable_cpu_data());
        }
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
        let b1 = bottom.get(1).map(|b| b.as_ref().borrow());
        let blobs = &self.layer.blobs;
        let bias = b1.unwrap_or_else(|| blobs[0].as_ref().borrow());
        let bias_data = bias.cpu_data();
        let mut t0 = top[0].borrow_mut();
        let top_data = t0.mutable_cpu_data();
        if !Rc::ptr_eq(&bottom[0], &top[0]) {
            let b0 = bottom[0].as_ref().borrow();
            let bottom_data = b0.cpu_data();
            caffe_copy(b0.count(), bottom_data, top_data);
        }

        let mut top_offset = 0usize;
        for n in 0..self.outer_dim {
            T::caffe_cpu_gemm(Transpose::None, Transpose::None, self.bias_dim, self.inner_dim, 1,
                              T::from_i32(1), bias_data, self.bias_multiplier.cpu_data(),
                              T::from_i32(1), &mut top_data[top_offset..]);
            top_offset += self.dim as usize;
        }
    }

    fn forward_gpu(&mut self, _bottom: &BlobVec<Self::DataType>, _top: &BlobVec<Self::DataType>) {
        no_gpu!();
    }

    fn backward_cpu(&mut self, top: &BlobVec<Self::DataType>, propagate_down: &Vec<bool>, bottom: &BlobVec<Self::DataType>) {
        if propagate_down[0] && !Rc::ptr_eq(&bottom[0], &top[0]) {
            let t0 = top[0].as_ref().borrow();
            let mut b0 = bottom[0].borrow_mut();
            let count = b0.count();
            caffe_copy(count, t0.cpu_diff(), b0.mutable_cpu_diff());
        }

        // in-place, we don't need to do anything with the data diff
        let bias_param = bottom.len() == 1;
        if (!bias_param && propagate_down[1]) || (bias_param && self.layer.param_propagate_down[0]) {
            let t0 = top[0].as_ref().borrow();
            let b1 = bottom.get(1).map(|b| b.borrow_mut());
            let blobs = &self.layer.blobs;
            let mut bias = b1.unwrap_or_else(|| blobs[0].borrow_mut());

            let top_diff = t0.cpu_diff();
            let bias_diff = bias.mutable_cpu_diff();
            let mut accum = bias_param;
            let mut top_offset = 0usize;
            for n in 0..self.outer_dim {
                T::caffe_cpu_gemv(Transpose::None, self.bias_dim, self.inner_dim, T::from_i32(1),
                                  &top_diff[top_offset..], self.bias_multiplier.cpu_data(),
                                  T::from_i32(accum as i32), bias_diff);
                top_offset += self.dim as usize;
                accum = true;
            }
        }
    }

    fn backward_gpu(&mut self, _top: &BlobVec<Self::DataType>, _propagate_down: &Vec<bool>, _bottom: &BlobVec<Self::DataType>) {
        no_gpu!();
    }
}

register_layer_class!(Bias);
