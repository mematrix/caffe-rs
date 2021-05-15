use std::rc::Rc;
use std::boxed::Box;
use std::cell::{RefCell, Ref, RefMut};

use protobuf::Clear;

use crate::common::{Caffe, CaffeBrew};
use crate::blob::{Blob, BlobType, BlobMemRef, BlobOp};
use crate::proto::caffe;
use crate::util::math_functions::{CaffeUtil, BlasOp, Blas};


/// A typedef for **shared_ptr** of `Blob<T>`.
pub type SharedBlob<T> = Rc<RefCell<Blob<T>>>;

/// A typedef of **vector of blob**.
pub type BlobVec<T> = Vec<SharedBlob<T>>;

/// A typedef for **shared_ptr** of `Layer<T>`.
pub type SharedLayer<T> = Rc<RefCell<Layer<T>>>;

/// A typedef of **vector of layer**.
pub type LayerVec<T> = Vec<SharedLayer<T>>;


#[derive(Clone, Default)]
pub struct LayerImpl<T: BlobType> {
    /// The protobuf that stores the layer parameters
    pub layer_param: caffe::LayerParameter,
    /// The phase: TRAIN or TEST
    pub phase: caffe::Phase,
    /// The vector that stores the learnable parameters as a set of blobs.
    pub blobs: BlobVec<T>,
    /// Vector indicating whether to compute the diff of each param blob.
    pub param_propagate_down: Vec<bool>,
    /// The vector that indicates whether each top blob has a non-zero weight in
    /// the objective function.
    pub loss: Vec<T>,
}

impl<T: BlobType> LayerImpl<T> {
    pub fn new(param: &caffe::LayerParameter) -> Self {
        let mut layer = LayerImpl {
            layer_param: param.clone(),
            ..Default::default()
        };

        // Set phase and copy blobs (if there are any).
        layer.phase = param.get_phase();
        if !layer.layer_param.get_blobs().is_empty() {
            layer.blobs.reserve(layer.layer_param.get_blobs().len());
            for x in layer.layer_param.get_blobs() {
                let mut blob = Blob::new();
                blob.set_from_proto(x, true);
                let blob = Rc::new(RefCell::new(blob));
                layer.blobs.push(blob);
            }
        }

        layer
    }
}


pub trait CaffeLayer<T: BlobType> {
    fn get_impl(&self) -> &LayerImpl<T>;

    fn get_impl_mut(&mut self) -> &mut LayerImpl<T>;

    fn layer_setup(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        def_layer_setup(self, bottom, top);
    }

    fn reshape(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>);

    fn to_proto(&self, param: &mut caffe::LayerParameter, write_diff: bool) {
        def_to_proto(self, param, write_diff);
    }

    fn layer_type(&self) -> &'static str {
        ""
    }

    fn exact_num_bottom_blobs(&self) -> i32 {
        -1
    }

    fn min_bottom_blobs(&self) -> i32 {
        -1
    }

    fn max_bottom_blobs(&self) -> i32 {
        -1
    }

    fn exact_num_top_blobs(&self) -> i32 {
        -1
    }

    fn min_top_blobs(&self) -> i32 {
        -1
    }

    fn max_top_blobs(&self) -> i32 {
        -1
    }

    fn equal_num_bottom_top_blobs(&self) -> bool {
        false
    }

    fn auto_top_blobs(&self) -> bool {
        false
    }

    fn allow_force_backward(&self, _bottom_index: usize) -> bool {
        true
    }

    fn forward_cpu(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>);

    fn forward_gpu(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        self.forward_cpu(bottom, top);
    }

    fn backward_cpu(&mut self, top: &BlobVec<T>, propagate_down: &Vec<bool>, bottom: &BlobVec<T>);

    fn backward_gpu(&mut self, top: &BlobVec<T>, propagate_down: &Vec<bool>, bottom: &BlobVec<T>) {
        self.backward_cpu(top, propagate_down, bottom);
    }

    fn check_blob_counts(&self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        def_check_blob_counts(self, bottom, top);
    }
}

pub fn def_layer_setup<T, Caffe>(_this: &mut Caffe, _bottom: &BlobVec<T>, _top: &BlobVec<T>)
    where T: BlobType, Caffe: CaffeLayer<T> + ?Sized {}

pub fn def_to_proto<T, Caffe>(this: &Caffe, param: &mut caffe::LayerParameter, write_diff: bool)
    where T: BlobType, Caffe: CaffeLayer<T> + ?Sized {
    param.clear();
    param.clone_from(&this.get_impl().layer_param);
    param.clear_blobs();
    for blob in &this.get_impl().blobs {
        RefCell::borrow(blob.as_ref()).to_proto(param.mut_blobs().push_default(), write_diff);
    }
}

pub fn def_check_blob_counts<T, Caffe>(this: &Caffe, bottom: &BlobVec<T>, top: &BlobVec<T>)
    where T: BlobType, Caffe: CaffeLayer<T> + ?Sized {
    if this.exact_num_bottom_blobs() >= 0 {
        let num = this.exact_num_bottom_blobs();
        check_eq!(num, bottom.len() as i32, "{} Layer takes {} bottom blob(s) as input.",
                  this.layer_type(), num);
    }
    if this.min_bottom_blobs() >= 0 {
        let num = this.min_bottom_blobs();
        check_le!(num, bottom.len() as i32, "{} Layer takes at least {} bottom blob(s) as input.",
                  this.layer_type(), num);
    }
    if this.max_bottom_blobs() >= 0 {
        let num = this.max_bottom_blobs();
        check_ge!(num, bottom.len() as i32, "{} Layer takes at most {} bottom blob(s) as input.",
                  this.layer_type(), num);
    }
    if this.exact_num_top_blobs() >= 0 {
        let num = this.exact_num_top_blobs();
        check_eq!(num, top.len() as i32, "{} Layer produces {} top blob(s) as output.",
                  this.layer_type(), num);
    }
    if this.min_top_blobs() >= 0 {
        let num = this.min_top_blobs();
        check_le!(num, top.len() as i32, "{} Layer produces at least {} top blob(s) as output.",
                  this.layer_type(), num);
    }
    if this.max_top_blobs() >= 0 {
        let num = this.max_top_blobs();
        check_ge!(num, top.len() as i32, "{} Layer produces at most {} top blob(s) as output.",
                  this.layer_type(), num);
    }
    if this.equal_num_bottom_top_blobs() {
        check_eq!(bottom.len(), top.len(),
                  "{} Layer produces one top blob as output for each bottom blob input.",
                  this.layer_type());
    }
}


pub struct Layer<T: BlobType> {
    layer: Box<dyn CaffeLayer<T>>,
}

impl<T: BlobType> Layer<T> {
    pub fn new(layer: Box<dyn CaffeLayer<T>>) -> Self {
        Layer {
            layer
        }
    }

    /// Implements common layer setup functionality.
    /// * `bottom`: the pre-shaped input blobs
    /// * `top`: the allocated but unshaped output blobs, to be shaped by Reshape
    ///
    /// Checks that the number of bottom and top blobs is correct.
    /// Calls [`layer_setup`][layer_setup] to do special layer setup for individual layer types,
    /// followed by [`reshape`][reshape] to set up sizes of top blobs and internal buffers.
    /// Sets up the loss weight multiplier blobs for any non-zero loss weights.
    /// This method may not be overridden.
    ///
    /// [layer_setup]: caffe_rs::Layer::layer_setup
    /// [reshape]: caffe_rs::Layer::reshape
    pub fn setup(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        self.layer.check_blob_counts(bottom, top);
        self.layer_setup(bottom, top);
        self.reshape(bottom, top);
        self.set_loss_weights(top);
    }

    pub fn layer_setup(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        self.layer.layer_setup(bottom, top);
    }

    pub fn reshape(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        self.layer.reshape(bottom, top);
    }

    pub fn forward(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) -> T {
        let mut loss = T::from_f32(0f32);
        self.reshape(bottom, top);

        match Caffe::mode() {
            CaffeBrew::CPU => {
                // CPU mode
                self.layer.forward_cpu(bottom, top);
                for top_id in 0..top.len() {
                    if self.loss(top_id).is_zero() {
                        continue;
                    }

                    let blob = RefCell::borrow(top[top_id].as_ref());
                    let count = blob.count();
                    let BlobMemRef { data, diff } = blob.cpu_mem_ref();
                    loss += Blas::<T>::caffe_cpu_dot(count as i32, data, diff);
                }
            }
            CaffeBrew::GPU => {
                self.layer.forward_gpu(bottom, top);
                unimplemented!();
            }
        }

        loss
    }

    pub fn backward(&mut self, top: &BlobVec<T>, propagate_down: &Vec<bool>, bottom: &BlobVec<T>) {
        match Caffe::mode() {
            CaffeBrew::CPU => {
                // CPU mode
                self.layer.backward_cpu(top, propagate_down, bottom);
            }
            CaffeBrew::GPU => {
                // GPU mode
                self.layer.backward_gpu(top, propagate_down, bottom);
            }
        }
    }

    pub fn blobs(&self) -> &BlobVec<T> {
        &self.layer.get_impl().blobs
    }

    pub fn blobs_mut(&mut self) -> &mut BlobVec<T> {
        &mut self.layer.get_impl_mut().blobs
    }

    pub fn layer_param(&self) -> &caffe::LayerParameter {
        &self.layer.get_impl().layer_param
    }

    pub fn to_proto(&self, param: &mut caffe::LayerParameter, write_diff: bool) {
        self.layer.to_proto(param, write_diff);
    }

    pub fn loss(&self, top_index: usize) -> T {
        let loss = &self.layer.get_impl().loss;
        if loss.len() > top_index {
            loss[top_index]
        } else {
            Default::default()
        }
    }

    pub fn set_loss(&mut self, top_index: usize, value: T) {
        let loss = &mut self.layer.get_impl_mut().loss;
        if loss.len() <= top_index {
            loss.resize(top_index + 1, Default::default());
        }

        loss[top_index] = value;
    }

    pub fn layer_type(&self) -> &'static str {
        self.layer.layer_type()
    }

    pub fn exact_num_bottom_blobs(&self) -> i32 {
        self.layer.exact_num_bottom_blobs()
    }

    pub fn min_bottom_blobs(&self) -> i32 {
        self.layer.min_bottom_blobs()
    }

    pub fn max_bottom_blobs(&self) -> i32 {
        self.layer.max_bottom_blobs()
    }

    pub fn exact_num_top_blobs(&self) -> i32 {
        self.layer.exact_num_top_blobs()
    }

    pub fn min_top_blobs(&self) -> i32 {
        self.layer.min_top_blobs()
    }

    pub fn max_top_blobs(&self) -> i32 {
        self.layer.max_top_blobs()
    }

    pub fn equal_num_bottom_top_blobs(&self) -> bool {
        self.layer.equal_num_bottom_top_blobs()
    }

    pub fn auto_top_blobs(&self) -> bool {
        self.layer.auto_top_blobs()
    }

    pub fn allow_force_backward(&self, bottom_index: usize) -> bool {
        self.layer.allow_force_backward(bottom_index)
    }

    pub fn param_propagate_down(&self, param_id: usize) -> bool {
        let prop_down = &self.layer.get_impl().param_propagate_down;
        if prop_down.len() > param_id {
            prop_down[param_id]
        } else {
            false
        }
    }

    pub fn set_param_propagate_down(&mut self, param_id: usize, value: bool) {
        let prop_down = &mut self.layer.get_impl_mut().param_propagate_down;
        if prop_down.len() <= param_id {
            prop_down.resize(param_id + 1, true);
        }

        prop_down[param_id] = value;
    }

    fn set_loss_weights(&mut self, top: &BlobVec<T>) {
        let num_loss_weights = self.layer.get_impl().layer_param.get_loss_weight().len();
        if num_loss_weights == 0usize {
            return;
        }

        check_eq!(top.len(), num_loss_weights, "loss_weight must be unspecified or specified once per top blob.");

        for top_id in 0..top.len() {
            let loss_weight = self.layer.get_impl().layer_param.get_loss_weight()[top_id];
            if loss_weight == 0f32 {
                continue;
            }

            self.set_loss(top_id, T::from_f32(loss_weight));
            let mut blob = top[top_id].borrow_mut();
            let count = blob.count();
            let loss_multiplier = blob.mutable_cpu_diff();
            CaffeUtil::caffe_set(count, T::from_f32(loss_weight), loss_multiplier);
        }
    }
}
