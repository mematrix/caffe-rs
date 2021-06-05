use crate::blob::BlobType;
use crate::layer::{LayerImpl, CaffeLayer, BlobVec};
use crate::proto::caffe::LayerParameter;
use crate::util::math_functions::caffe_set;


/// Index into the input blob along its first axis.
///
/// This layer can be used to select, reorder, and even replicate examples in a
/// batch.  The second blob is cast to int and treated as an index into the
/// first axis of the first blob.
pub struct BatchReindexLayer<T: BlobType> {
    layer: LayerImpl<T>,
}

impl<T: BlobType> BatchReindexLayer<T> {
    pub fn new(param: &LayerParameter) -> Self {
        Self {
            layer: LayerImpl::new(param),
        }
    }

    fn check_batch_reindex(&self, initial_num: i32, final_num: usize, ridx_data: &[T]) {
        assert!(final_num <= ridx_data.len());
        for i in 0..final_num {
            let d = *unsafe { ridx_data.get_unchecked(i) };
            assert!(d >= T::default(), "Index specified for reindex layer was negative.");
            assert!(d < T::from_i32(initial_num), "Index specified for reindex layer was greater than batch size.");
        }
    }
}

impl<T: BlobType> CaffeLayer for BatchReindexLayer<T> {
    type DataType = T;

    fn get_impl(&self) -> &LayerImpl<Self::DataType> {
        &self.layer
    }

    fn get_impl_mut(&mut self) -> &mut LayerImpl<Self::DataType> {
        &mut self.layer
    }

    fn reshape(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        let b0 = bottom[0].as_ref().borrow();
        let b1 = bottom[1].as_ref().borrow();
        assert_eq!(1, b1.num_axes());

        let shape = b0.shape();
        let mut new_shape = shape.clone();
        if new_shape.is_empty() {
            new_shape.push(b1.shape_idx(0));
        } else {
            new_shape[0] = b1.shape_idx(0);
        }

        top[0].borrow_mut().reshape(&new_shape);
    }

    fn layer_type(&self) -> &'static str {
        "BatchReindex"
    }

    fn exact_num_bottom_blobs(&self) -> i32 {
        2
    }

    fn exact_num_top_blobs(&self) -> i32 {
        1
    }

    fn forward_cpu(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        let b0 = bottom[0].as_ref().borrow();
        let b1 = bottom[1].as_ref().borrow();
        self.check_batch_reindex(b0.shape_idx(0), b1.count(), b1.cpu_data());

        let mut t0 = top[0].borrow_mut();
        let t0_count = t0.count();
        if t0_count == 0 {
            return;
        }

        let inner_dim = b0.count() / b0.shape_idx(0) as usize;
        let d_in = b0.cpu_data();
        let permut = b1.cpu_data();
        let out = t0.mutable_cpu_data();
        for index in 0..t0_count {
            let n = index / inner_dim;
            let in_n = permut[n].to_usize();
            out[index] = d_in[in_n * inner_dim + index % inner_dim];
        }
    }

    fn forward_gpu(&mut self, _bottom: &BlobVec<Self::DataType>, _top: &BlobVec<Self::DataType>) {
        no_gpu!();
    }

    fn backward_cpu(&mut self, top: &BlobVec<Self::DataType>, propagate_down: &Vec<bool>, bottom: &BlobVec<Self::DataType>) {
        assert!(!propagate_down[1], "Cannot backprop to index.");
        if !propagate_down[0] {
            return;
        }

        let mut b0 = bottom[0].borrow_mut();
        let b1 = bottom[1].as_ref().borrow();
        let t0 = top[0].as_ref().borrow();
        let count = b0.count();
        let inner_dim = count / b0.shape_idx(0) as usize;
        let bot_diff = b0.mutable_cpu_diff();
        let permut = b1.cpu_data();
        let top_diff = t0.cpu_diff();
        caffe_set(count, T::default(), bot_diff);
        for index in 0..t0.count() {
            let n = index / inner_dim;
            let in_n = permut[n].to_usize();
            bot_diff[in_n * inner_dim + index % inner_dim] += top_diff[index];
        }
    }

    fn backward_gpu(&mut self, _top: &BlobVec<Self::DataType>, _propagate_down: &Vec<bool>, _bottom: &BlobVec<Self::DataType>) {
        no_gpu!();
    }
}

register_layer_class!(BatchReindex);
