use std::rc::Rc;
use std::boxed::Box;
use std::cell::{RefCell, Ref, RefMut};

use crate::proto::caffe;
use crate::blob::{Blob, BlobType};
use crate::util::math_functions::CaffeUtil;


#[derive(Clone, Default)]
pub struct BlobVec<T: BlobType>(Vec<Rc<RefCell<Blob<T>>>>);

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
            layer.blobs.0.reserve(layer.layer_param.get_blobs().len());
            for x in layer.layer_param.get_blobs() {
                let mut blob = Blob::new();
                blob.set_from_proto(x, true);
                let blob = Rc::new(RefCell::new(blob));
                layer.blobs.0.push(blob);
            }
        }

        layer
    }
}


pub trait CaffeLayer<T: BlobType> {
    fn get_impl(&self) -> &LayerImpl<T>;

    fn get_impl_mut(&mut self) -> &mut LayerImpl<T>;

    fn layer_setup(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {}

    fn reshape(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>);

    fn to_proto(&self, param: &mut caffe::LayerParameter, write_diff: bool) {
        //
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

    fn allow_force_backward(&self, _bottom_index: i32) -> bool {
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
        if self.exact_num_bottom_blobs() >= 0 {
            let num = self.exact_num_bottom_blobs();
            check_eq!(num, bottom.0.len() as i32, "{} Layer takes {} bottom blob(s) as input.",
                      self.layer_type(), num);
        }
        if self.min_bottom_blobs() >= 0 {
            let num = self.min_bottom_blobs();
            check_le!(num, bottom.0.len() as i32, "{} Layer takes at least {} bottom blob(s) as input.",
                      self.layer_type(), num);
        }
        if self.max_bottom_blobs() >= 0 {
            let num = self.max_bottom_blobs();
            check_ge!(num, bottom.0.len() as i32, "{} Layer takes at most {} bottom blob(s) as input.",
                      self.layer_type(), num);
        }
        if self.exact_num_top_blobs() >= 0 {
            let num = self.exact_num_top_blobs();
            check_eq!(num, top.0.len() as i32, "{} Layer produces {} top blob(s) as output.",
                      self.layer_type(), num);
        }
        if self.min_top_blobs() >= 0 {
            let num = self.min_top_blobs();
            check_le!(num, top.0.len() as i32, "{} Layer produces at least {} top blob(s) as output.",
                      self.layer_type(), num);
        }
        if self.max_top_blobs() >= 0 {
            let num = self.max_top_blobs();
            check_ge!(num, top.0.len() as i32, "{} Layer produces at most {} top blob(s) as output.",
                      self.layer_type(), num);
        }
        if self.equal_num_bottom_top_blobs() {
            check_eq!(bottom.0.len(), top.0.len(),
                      "{} Layer produces one top blob as output for each bottom blob input.",
                      self.layer_type());
        }
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

    pub fn setup(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        // todo
    }

    pub fn layer_setup(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        self.layer.layer_setup(bottom, top);
    }

    pub fn reshape(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        self.layer.reshape(bottom, top);
    }

    pub fn forward(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) -> T {
        let loss = T::from_f32(0f32);
        self.reshape(bottom, top);

        // CPU mode
        self.layer.forward_cpu(bottom, top);
        // todo: for

        loss
    }

    pub fn backward(&mut self, top: &BlobVec<T>, propagate_down: &Vec<bool>, bottom: &BlobVec<T>) {
        //
    }

    pub fn blobs(&mut self) -> &mut BlobVec<T> {
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

    pub fn allow_force_backward(&self, bottom_index: i32) -> bool {
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

        check_eq!(top.0.len(), num_loss_weights, "loss_weight must be unspecified or specified once per top blob.");

        for top_id in 0..top.0.len() {
            let loss_weight = self.layer.get_impl().layer_param.get_loss_weight()[top_id];
            if loss_weight == 0f32 {
                continue;
            }

            self.set_loss(top_id, T::from_f32(loss_weight));
            let mut blob = top.0[top_id].borrow_mut();
            let count = blob.count();
            let loss_multiplier = blob.mutable_cpu_diff();
            CaffeUtil::caffe_set(count, T::from_f32(loss_weight), loss_multiplier);
        }
    }
}
