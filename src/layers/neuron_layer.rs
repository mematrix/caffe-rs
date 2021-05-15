use crate::blob::{BlobType};
use crate::layer::{LayerImpl, BlobVec};
use crate::proto::caffe::{LayerParameter};


/// An interface for layers that take one blob as input ($ x $) and produce one
/// equally-sized blob as output ($ y $), where each element of the output
/// depends only on the corresponding input element.
pub struct NeuronLayer<T: BlobType> {
    layer: LayerImpl<T>,
}

impl<T: BlobType> NeuronLayer<T> {
    pub fn new(param: &LayerParameter) -> Self {
        NeuronLayer {
            layer: LayerImpl::new(param)
        }
    }

    pub fn get_impl(&self) -> &LayerImpl<T> {
        &self.layer
    }

    pub fn get_impl_mut(&mut self) -> &mut LayerImpl<T> {
        &mut self.layer
    }

    pub fn reshape(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        top[0].borrow_mut().reshape_like(&*bottom[0].as_ref().borrow());
    }

    pub fn exact_num_bottom_blobs(&self) -> i32 {
        1
    }

    pub fn exact_num_top_blobs(&self) -> i32 {
        1
    }
}
