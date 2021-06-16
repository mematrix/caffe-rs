use crate::blob::{BlobType, MAX_BLOB_AXES};
use crate::layer::{LayerImpl, CaffeLayer, BlobVec, def_layer_setup};
use crate::proto::caffe::LayerParameter;
use crate::util::math_functions::caffe_copy;


/// Takes at least two `Blob`s and concatenates them along either the num
/// or channel dimension, outputting the result.
pub struct ConcatLayer<T: BlobType> {
    layer: LayerImpl<T>,
    count: i32,
    num_concats: i32,
    concat_input_size: i32,
    concat_axis: i32,
}

impl<T: BlobType> ConcatLayer<T> {
    pub fn new(param: &LayerParameter) -> Self {
        Self {
            layer: LayerImpl::new(param),
            count: 0,
            num_concats: 0,
            concat_input_size: 0,
            concat_axis: 0,
        }
    }
}

impl<T: BlobType> CaffeLayer for ConcatLayer<T> {
    type DataType = T;

    fn get_impl(&self) -> &LayerImpl<Self::DataType> {
        &self.layer
    }

    fn get_impl_mut(&mut self) -> &mut LayerImpl<Self::DataType> {
        &mut self.layer
    }

    fn layer_setup(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        let concat_param = self.layer.layer_param.get_concat_param();
        assert!(!(concat_param.has_axis() && concat_param.has_concat_dim()),
                "Either axis or concat_dim should be specified; not both.");
    }

    fn reshape(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        let b0 = bottom[0].as_ref().borrow();
        let num_axes = b0.num_axes();
        let concat_param = self.layer.layer_param.get_concat_param();
        if concat_param.has_concat_dim() {
            self.concat_axis = concat_param.get_concat_dim() as i32;
            assert!(self.concat_axis >= 0, "casting concat_dim from uint32 to int32 produced \
                    negative result; concat_dim must satisfy 0 <= concat_dim < {}", MAX_BLOB_AXES);
            assert!(self.concat_axis < num_axes, "concat_dim out of range.");
        } else {
            self.concat_axis = b0.canonical_axis_index(concat_param.get_axis()) as i32;
        }

        // Initialize with the first blob.
        let mut top_shape = b0.shape().clone();
        self.num_concats = b0.count_range(0, self.concat_axis as usize);
        self.concat_input_size = b0.count_range_to_end((self.concat_axis + 1) as usize);
        let mut bottom_count_sum = b0.count();
        for i in 1..bottom.len() {
            let bi = bottom[i].as_ref().borrow();
            assert_eq!(num_axes, bi.num_axes(), "All inputs must have the same #axes.");
            for j in 0..num_axes {
                if j == self.concat_axis { continue; }
                assert_eq!(top_shape[j as usize], bi.shape_idx(j),
                           "All inputs must have the same shape, except at concat_axis.");
            }

            bottom_count_sum += bi.count();
            top_shape[self.concat_axis as usize] += bi.shape_idx(self.concat_axis);
        }

        let mut t0 = top[0].borrow_mut();
        let top_count = t0.count();
        t0.reshape(&top_shape);
        assert_eq!(bottom_count_sum, top_count);
        if bottom.len() == 1 {
            t0.share_data(&*b0);
            t0.share_diff(&*b0);
        }
    }

    fn layer_type(&self) -> &'static str {
        "Concat"
    }

    fn min_bottom_blobs(&self) -> i32 {
        1
    }

    fn exact_num_top_blobs(&self) -> i32 {
        1
    }

    fn forward_cpu(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        if bottom.len() == 1 {
            return;
        }

        let mut t0 = top[0].borrow_mut();
        let top_concat_axis = t0.shape_idx(self.concat_axis);
        let top_data = t0.mutable_cpu_data();
        let mut offset_concat_axis = 0;
        for bi in bottom {
            let blob = bi.as_ref().borrow();
            let bottom_data = blob.cpu_data();
            let bottom_concat_axis = blob.shape_idx(self.concat_axis);
            for n in 0..self.num_concats {
                let size = (bottom_concat_axis * self.concat_input_size) as usize;
                let top_offset = ((n * top_concat_axis + offset_concat_axis) * self.concat_input_size) as usize;
                caffe_copy(size, &bottom_data[(n as usize * size)..], &mut top_data[top_offset..]);
            }
            offset_concat_axis += bottom_concat_axis;
        }
    }

    fn forward_gpu(&mut self, _bottom: &BlobVec<Self::DataType>, _top: &BlobVec<Self::DataType>) {
        no_gpu!();
    }

    fn backward_cpu(&mut self, top: &BlobVec<Self::DataType>, propagate_down: &Vec<bool>,
                    bottom: &BlobVec<Self::DataType>) {
        if bottom.len() == 1 {
            return;
        }

        let t0 = top[0].as_ref().borrow();
        let top_diff = t0.cpu_diff();
        let mut offset_concat_axis = 0;
        let top_concat_axis = t0.shape_idx(self.concat_axis);
        for i in 0..bottom.len() {
            let mut blob = bottom[i].borrow_mut();
            let bottom_concat_axis = blob.shape_idx(self.concat_axis);
            if propagate_down[i] {
                let bottom_diff = blob.mutable_cpu_diff();
                for n in 0..self.num_concats {
                    let size = (bottom_concat_axis * self.concat_input_size) as usize;
                    let top_offset = ((n * top_concat_axis + offset_concat_axis) * self.concat_input_size) as usize;
                    caffe_copy(size, &top_diff[top_offset..], &mut bottom_diff[(n as usize * size)..]);
                }
            }
            offset_concat_axis += bottom_concat_axis;
        }
    }

    fn backward_gpu(&mut self, _top: &BlobVec<Self::DataType>, _propagate_down: &Vec<bool>,
                    _bottom: &BlobVec<Self::DataType>) {
        no_gpu!();
    }
}

register_layer_class!(Concat);
