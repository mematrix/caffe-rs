use std::cmp::Ordering;

use crate::blob::BlobType;
use crate::layer::{CaffeLayer, LayerImpl, BlobVec};
use crate::proto::caffe::LayerParameter;


/// Compute the index of the $ K $ max values for each datum across all dimensions
/// $ (C \times H \times W) $.
///
/// Intended for use after a classification layer to produce a prediction. If parameter
/// out_max_val is set to true, output is a vector of pairs (max_ind, max_val) for each
/// image. The axis parameter specifies an axis along which to maximise.
///
/// **NOTE**: **does not** implement `Backwards` operation.
pub struct ArgMaxLayer<T: BlobType> {
    layer: LayerImpl<T>,
    out_max_val: bool,
    top_k: usize,
    has_axis: bool,
    axis: i32,
}

impl<T: BlobType> ArgMaxLayer<T> {
    /// `param` provides **ArgMaxParameter** argmax_param, with **ArgMaxLayer options**:
    /// - top_k (**optional uint, default `1`**). the number $ K $ of maximal items to output.
    /// - out_max_val (**optional bool, default `false`**). if set, output a vector of pairs
    /// (max_ind, max_val) unless axis is set then output max_val along the specified axis.
    /// - axis (**optional int**). if set, maximise along the specified axis else maximise
    /// the flattened trailing dimensions for each index of the first / num dimension.
    pub fn new(param: &LayerParameter) -> Self {
        ArgMaxLayer {
            layer: LayerImpl::new(param),
            out_max_val: false,
            top_k: 0,
            has_axis: false,
            axis: 0,
        }
    }
}

impl<T: BlobType> CaffeLayer for ArgMaxLayer<T> {
    type DataType = T;

    fn get_impl(&self) -> &LayerImpl<T> {
        &self.layer
    }

    fn get_impl_mut(&mut self) -> &mut LayerImpl<T> {
        &mut self.layer
    }

    fn layer_setup(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        let argmax_param = self.layer.layer_param.get_argmax_param();
        self.out_max_val = argmax_param.get_out_max_val();
        self.top_k = argmax_param.get_top_k() as usize;
        self.has_axis = argmax_param.has_axis();
        check_ge!(self.top_k, 1, "top k must not be less than 1.");

        if self.has_axis {
            let b0 = bottom[0].as_ref().borrow();
            self.axis = b0.canonical_axis_index(argmax_param.get_axis()) as i32;
            check_ge!(self.axis, 0, "axis must not be less than 0.");
            check_le!(self.axis, b0.num_axes(), "axis must be less than or equal to the number of axis.");
            check_le!(self.top_k as i32, b0.shape_idx(self.axis),
                "top_k must be less than or equal to the dimension of the axis.");
        } else {
            let count = bottom[0].as_ref().borrow().count_range_to_end(1);
            check_le!(self.top_k as i32, count, "top_k must be less than or equal to the \
                dimension of the flattened bottom blob per instance.");
        }
    }

    fn reshape(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        let b0 = bottom[0].as_ref().borrow();
        let mut num_top_axes = b0.num_axes();
        if num_top_axes < 3 {
            num_top_axes = 3;
        }

        let mut shape;
        if self.has_axis {
            // Produces max_ind or max_val per axis
            shape = b0.shape().clone();
            shape[self.axis as usize] = self.top_k as i32;
        } else {
            let num_top_axes = num_top_axes as usize;
            shape = Vec::with_capacity(num_top_axes);
            shape.resize(num_top_axes, 1);
            shape[0] = b0.shape_idx(0);
            // Produces max_ind
            shape[2] = self.top_k as i32;
            if self.out_max_val {
                // Produces max_ind and max_val
                shape[1] = 2;
            }
        }

        top[0].borrow_mut().reshape(&shape);
    }

    fn layer_type(&self) -> &'static str {
        "ArgMax"
    }

    fn exact_num_bottom_blobs(&self) -> i32 {
        1
    }

    fn exact_num_top_blobs(&self) -> i32 {
        1
    }

    /// `bottom` input Blob vector (length 1).
    /// - $ (N \times C \times H \times W) $ the inputs $ x $.
    ///
    /// `top` output Blob vector (length 1).
    /// - $ (N \times 1 \times K) $ or, if `out_max_val` $ (N \times 2 \times K) $ unless `axis`
    /// is set than e.g. $ (N \times K \times H \times W) $ if `axis == 1` the computed outputs
    /// $$ y\_n = \arg\max\limits\_i x\_{ni} $$ (for $ K = 1 $).
    fn forward_cpu(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        let b0 = bottom[0].as_ref().borrow();
        let mut t0 = top[0].borrow_mut();
        let bottom_data = b0.cpu_data();
        let top_data = t0.mutable_cpu_data();
        let dim;
        let axis_dist;
        if self.has_axis {
            dim = b0.shape_idx(self.axis);
            // Distance between values of axis in blob
            axis_dist = b0.count_range_to_end(self.axis as usize) / dim;
        } else {
            dim = b0.count_range_to_end(1);
            axis_dist = 1;
        }

        let dim = dim as usize;
        let axis_dist = axis_dist as usize;
        let num = b0.count() / dim;
        let mut bottom_data_vector: Vec<(T, usize)> = Vec::with_capacity(dim);
        bottom_data_vector.resize(dim, Default::default());
        let cmp = |a: &(T, usize), b: &(T, usize)| {
            // Treat the `NAN` as the minimum value.
            b.partial_cmp(a).unwrap_or_else(|| if (*a).0.is_nan_v() { Ordering::Greater } else { Ordering::Less })
        };
        for i in 0..num {
            for j in 0..dim {
                bottom_data_vector[j] = (bottom_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
            }
            // C++ std::partial_sort
            let (first, _, _) = bottom_data_vector.select_nth_unstable_by(self.top_k, &cmp);
            first.sort_unstable_by(&cmp);

            for j in 0..self.top_k {
                if self.out_max_val {
                    if self.has_axis {
                        // Produces max_val per axis
                        top_data[(i / axis_dist * self.top_k + j) * axis_dist + i % axis_dist] =
                            bottom_data_vector[j].0;
                    } else {
                        // Produces max_ind and max_val
                        top_data[2usize * i * self.top_k + j] = T::from_usize(bottom_data_vector[j].1);
                        top_data[2usize * i * self.top_k + self.top_k + j] = bottom_data_vector[j].0;
                    }
                } else {
                    // Produces max_ind per axis
                    top_data[(i / axis_dist * self.top_k + j) * axis_dist + i % axis_dist] =
                        T::from_usize(bottom_data_vector[j].1);
                }
            }
        }
    }

    /// Not implemented (non-differentiable function).
    fn backward_cpu(&mut self, _top: &BlobVec<T>, _propagate_down: &Vec<bool>, _bottom: &BlobVec<T>) {
        unimplemented!();
    }
}

register_layer_class!(ArgMax);
