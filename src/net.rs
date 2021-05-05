use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::option::Option::Some;
use std::rc::Rc;

use crate::common::{Caffe, CaffeBrew};
use crate::blob::{Blob, BlobOp, BlobType};
use crate::layer::{Layer, SharedBlob, BlobVec, SharedLayer, LayerVec};
use crate::layer_factory::{LayerRegister, LayerRegistry};
use crate::proto::caffe::{Phase, NetParameter, NetState, NetStateRule, ParamSpec, ParamSpec_DimCheckMode};
use crate::util::insert_splits::{insert_splits};
use protobuf::{Chars, Clear};
use crate::util::math_functions::CaffeUtil;


/// Callback invoked at specific points during at iteration.
pub trait NetCallback {
    fn run(&mut self, layer: usize);
}

/// Connect [Layer][layer] together into a directed acyclic graph (DAG)
/// specified by a `NetParameter`.
///
/// [layer]: caffe_rs::layer::Layer
#[derive(Default)]
pub struct Net<T: BlobType> {
    /// The network name.
    name: String,
    /// The phase: `TRAIN` or `TEST`.
    phase: Phase,

    /// Individual layers in the net
    layers: LayerVec<T>,
    layer_names: Vec<String>,
    layer_names_index: HashMap<String, usize>,
    layer_need_backward: Vec<bool>,

    /// The blobs storing intermediate results between the layer.
    blobs: BlobVec<T>,
    blob_names: Vec<String>,
    blob_names_index: HashMap<String, usize>,
    blob_need_backward: Vec<bool>,

    /// `bottom_vecs` stores the vectors containing the input for each layer.
    /// They don't actually host the blobs (`blobs` does).
    bottom_vecs: Vec<BlobVec<T>>,
    bottom_id_vecs: Vec<Vec<usize>>,
    bottom_need_backward: Vec<Vec<bool>>,

    /// `top_vecs` stores the vectors containing the output for each layer.
    top_vecs: Vec<BlobVec<T>>,
    top_id_vecs: Vec<Vec<usize>>,

    /// Vector of weight in the loss (or objective) function of each net blob, indexed by blob_id.
    blob_loss_weights: Vec<T>,
    param_id_vecs: Vec<Vec<usize>>,
    param_owners: Vec<i32>,
    param_display_names: Vec<String>,
    param_layer_indices: Vec<(usize, usize)>,
    param_names_index: HashMap<String, usize>,

    /// blob indices for the input of the net.
    net_input_blob_indices: Vec<usize>,
    /// blob indices for the output of the net.
    net_output_blob_indices: Vec<usize>,
    net_input_blobs: BlobVec<T>,
    net_output_blobs: BlobVec<T>,

    /// The parameters in the network.
    params: BlobVec<T>,
    learnable_params: BlobVec<T>,
    /// The mapping from `params` -> `learnable_params`: we have
    /// `learnable_param_ids.len() == params.len()`, and
    /// `learnable_params[learnable_param_ids[i]] == params[i]` if and only if `params[i]`
    /// is an "**owner**"; otherwise, `params[i]` is a sharer and
    /// `learnable_params[learnable_param_ids[i]]` gives its owner.
    learnable_param_ids: Vec<usize>,
    /// The learning rate multipliers for `learnable_params`.
    params_lr: Vec<f32>,
    has_params_lr: Vec<bool>,
    /// The weight decay multipliers for `learnable_params`.
    params_weight_decay: Vec<f32>,
    has_params_decay: Vec<bool>,

    /// The bytes of memory used by this net.
    memory_used: usize,
    /// Whether to compute and display debug info for the net.
    debug_info: bool,

    // Callbacks
    before_forward: Vec<Box<dyn NetCallback>>,
    after_forward: Vec<Box<dyn NetCallback>>,
    before_backward: Vec<Box<dyn NetCallback>>,
    after_backward: Vec<Box<dyn NetCallback>>,
}

impl<T: BlobType> Net<T> {
    pub fn new(param: &NetParameter) -> Self {
        let mut net: Self = Default::default();
        net.init(param);
        net
    }

    pub fn from_file(param_file: &str, phase: Phase, level: i32, stages: &Option<Vec<String>>) -> Self {
        let mut param = NetParameter::new();
        // todo: init param content. --> ReadNetParamsFromTextFileOrDie

        param.mut_state().set_phase(phase);
        if let Some(ref stages) = *stages {
            let param_stage = param.mut_state().mut_stage();
            for x in stages {
                param_stage.push(Chars::from(x.as_str()));
            }
        }
        param.mut_state().set_level(level);

        let mut net: Self = Default::default();
        net.init(&param);
        net
    }

    pub fn init(&mut self, in_param: &NetParameter) {
        // set phase from the state.
        self.phase = in_param.get_state().get_phase();
        // filter layers based on their include/exclude rules and the current NetState.
        let mut filtered_param = NetParameter::new();
        Self::filter_net(in_param, &mut filtered_param);
        if Caffe::root_solver() {
            info!("Initializing net from parameters:\n{:?}", filtered_param);
        }

        // create a copy of filtered_param with splits added where necessary.
        let mut param = NetParameter::new();
        insert_splits(&filtered_param, &mut param);

        self.name = param.get_name().to_string();
        self.memory_used = 0usize;
        // for each layer, set up its input and output.
        let layer_size = param.get_layer().len();
        self.bottom_vecs.resize(layer_size, BlobVec::new());
        self.top_vecs.resize(layer_size, BlobVec::new());
        self.bottom_id_vecs.resize(layer_size, Default::default());
        self.param_id_vecs.resize(layer_size, Default::default());
        self.top_id_vecs.resize(layer_size, Default::default());
        self.bottom_need_backward.resize(layer_size, Default::default());

        // basically, build all the layers and set up their connections.
        let mut blob_name_to_idx = Some(HashMap::new());
        let mut available_blobs = Some(HashSet::new());

        for layer_id in 0..layer_size {
            // inherit phase from net if unset.
            if !param.get_layer()[layer_id].has_phase() {
                param.mut_layer()[layer_id].set_phase(self.phase);
            }
            // setup layer.
            let layer_param = &param.get_layer()[layer_id];
            if !layer_param.get_propagate_down().is_empty() {
                check_eq!(layer_param.get_propagate_down().len(), layer_param.get_bottom().len(),
                          "propagate_down param must be specified either 0 or bottom_size times");
            }
            self.layers.push(LayerRegistry::create_layer(layer_param));
            self.layer_names.push(layer_param.get_name().to_string());
            if Caffe::root_solver() {
                info!("Creating Layer {:?}", layer_param.get_name());
            }

            // figure out this layer's input and output
            let mut need_backward = false;
            for bottom_id in 0..layer_param.get_bottom().len() {
                let blob_id = self.append_bottom(&param, layer_id, bottom_id,
                                                 available_blobs.as_mut().unwrap(),
                                                 blob_name_to_idx.as_mut().unwrap());
                // if a blob needs backward this layer should provide it.
                need_backward |= self.blob_need_backward[blob_id];
            }
            let mut num_top = layer_param.get_top().len();
            for top_id in 0..num_top {
                self.append_top(&param, layer_id, top_id, &mut available_blobs, &mut blob_name_to_idx);
                // collect Input layer tops as Net inputs.
                if layer_param.get_field_type() == "Input" {
                    let blob_id = self.blobs.len() - 1usize;
                    self.net_input_blob_indices.push(blob_id);
                    self.net_input_blobs.push(self.blobs[blob_id].clone());
                }
            }
            // if the layer specifies that AutoTopBlobs() -> true and the LayerParameter
            // specified fewer than the required number (as specified by ExactNubTopBlobs()
            // or MinTopBlobs()), allocate them here.
            if self.layers[layer_id].as_ref().borrow().auto_top_blobs() {
                let needed_num_top;
                {
                    let layer = self.layers[layer_id].as_ref().borrow();
                    needed_num_top = std::cmp::max(layer.min_top_blobs(), layer.exact_num_top_blobs());
                }
                if needed_num_top >= 0 {
                    let needed_num_top = needed_num_top as usize;
                    while num_top < needed_num_top {
                        // add "anonymous" top blobs, -- do not modify available_blobs or
                        // blob_name_to_idx as we don't want there blobs to be usable as input
                        // to other layers.
                        self.append_top(&param, layer_id, num_top, &mut Option::None, &mut Option::None);
                        num_top += 1;
                    }
                }
            }
            // after this layer is connected, set it up.
            self.layers[layer_id].as_ref().borrow_mut().setup(&self.bottom_vecs[layer_id], &self.top_vecs[layer_id]);
            if Caffe::root_solver() {
                info!("Setting up {:?}", self.layer_names[layer_id]);
            }
            for top_id in 0..self.top_vecs[layer_id].len() {
                let blob_id = self.top_id_vecs[layer_id][top_id];
                if self.blob_loss_weights.len() <= blob_id {
                    self.blob_loss_weights.resize(blob_id + 1, T::default());
                }

                let loss = self.layers[layer_id].as_ref().borrow().loss(top_id);
                self.blob_loss_weights[blob_id] = loss;
                if Caffe::root_solver() {
                    info!("Top shape: {:?}", self.top_vecs[layer_id][top_id].as_ref().borrow().shape_string());
                }
                if !loss.is_zero() {
                    if Caffe::root_solver() {
                        info!("    with loss weight {:?}", loss);
                    }
                }

                self.memory_used += self.top_vecs[layer_id][top_id].as_ref().borrow().count();
            }
            if Caffe::root_solver() {
                info!("Memory required for data: {:?}", self.memory_used * std::mem::size_of::<T>());
            }

            let param_size = layer_param.get_param().len();
            let num_param_blobs = self.layers[layer_id].as_ref().borrow().blobs().len();
            check_le!(param_size, num_param_blobs, "Too many params specified for layer {:?}", layer_param.get_name());
            let default_param_spec = ParamSpec::new();
            for param_id in 0..num_param_blobs {
                let param_spec = if param_id < param_size {
                    &layer_param.get_param()[param_id]
                } else {
                    &default_param_spec
                };
                let param_need_backward = param_spec.get_lr_mult() != 0f32;
                need_backward |= param_need_backward;
                self.layers[layer_id].borrow_mut().set_param_propagate_down(param_id, param_need_backward);
            }
            for param_id in 0..num_param_blobs {
                self.append_param(&param, layer_id, param_id);
            }

            // finally, set the backward flag
            self.layer_need_backward.push(need_backward);
            if need_backward {
                for &top_id in &self.top_id_vecs[layer_id] {
                    self.blob_need_backward[top_id] = true;
                }
            }
        }

        // Go through the net backwards to determine which blobs contribute to the loss.
        // We can skip backward computation for blobs that don't contribute to the loss.
        // Also checks if all bottom blobs don't need backward computation (possible because
        // the skip_propagate_down param) and so we can skip backward computation for the entire layer
        let mut blobs_under_loss = HashSet::new();
        let mut blobs_skip_back_prop = HashSet::new();
        for layer_id in (0..self.layers.len()).rev() {
            let mut layer_contributes_loss = false;
            let mut layer_skip_propagate_down = true;
            {
                let layer_loss = self.layers[layer_id].as_ref().borrow();
                for top_id in 0..self.top_vecs.len() {
                    let blob_name = &self.blob_names[self.top_id_vecs[layer_id][top_id]];
                    if !layer_loss.loss(top_id).is_zero() || blobs_under_loss.contains(blob_name) {
                        layer_contributes_loss = true;
                    }
                    if !blobs_skip_back_prop.contains(blob_name) {
                        layer_skip_propagate_down = false;
                    }
                    if layer_contributes_loss && !layer_skip_propagate_down {
                        break;
                    }
                }
            }
            // If this layer can skip backward computation, also all his bottom blobs don't
            // need backpropagation
            if self.layer_need_backward[layer_id] && layer_skip_propagate_down {
                self.layer_need_backward[layer_id] = false;
                for bottom_id in 0..self.bottom_vecs[layer_id].len() {
                    self.bottom_need_backward[layer_id][bottom_id] = false;
                }
            }
            if !layer_contributes_loss {
                self.layer_need_backward[layer_id] = false;
            }
            if Caffe::root_solver() {
                if self.layer_need_backward[layer_id] {
                    info!("{:?} needs backward computation.", self.layer_names[layer_id]);
                } else {
                    info!("{:?} does not need backward computation.", self.layer_names[layer_id]);
                }
            }

            for bottom_id in 0..self.bottom_vecs[layer_id].len() {
                if layer_contributes_loss {
                    let blob_name = &self.blob_names[self.bottom_id_vecs[layer_id][bottom_id]];
                    blobs_under_loss.insert(blob_name.clone());
                } else {
                    self.bottom_need_backward[layer_id][bottom_id] = false;
                }

                if !self.bottom_need_backward[layer_id][bottom_id] {
                    let blob_name = &self.blob_names[self.bottom_id_vecs[layer_id][bottom_id]];
                    blobs_skip_back_prop.insert(blob_name.clone());
                }
            }
        }

        // Handle force_backward if needed
        if param.get_force_backward() {
            for layer_id in 0..self.layers.len() {
                self.layer_need_backward[layer_id] = true;
                for bottom_id in 0.. self.bottom_need_backward[layer_id].len() {
                    self.bottom_need_backward[layer_id][bottom_id] =
                        self.bottom_need_backward[layer_id][bottom_id] ||
                            self.layers[layer_id].as_ref().borrow().allow_force_backward(bottom_id);
                    self.blob_need_backward[self.bottom_id_vecs[layer_id][bottom_id]] =
                        self.blob_need_backward[self.bottom_id_vecs[layer_id][bottom_id]] ||
                            self.bottom_need_backward[layer_id][bottom_id];
                }

                for param_id in 0..self.layers[layer_id].as_ref().borrow().blobs().len() {
                    self.layers[layer_id].borrow_mut().set_param_propagate_down(param_id, true);
                }
            }
        }

        // In the end, all remaining blobs are considered output blobs.
        let available_blobs = available_blobs.unwrap();
        let blob_name_to_idx = blob_name_to_idx.unwrap();
        for it in &available_blobs {
            if Caffe::root_solver() {
                info!("This network produces output {:?}", it);
            }

            let idx = blob_name_to_idx[it];
            self.net_output_blobs.push(self.blobs[idx].clone());
            self.net_output_blob_indices.push(idx);
        }
        for blob_id in 0..self.blob_names.len() {
            self.blob_names_index.insert(self.blob_names[blob_id].clone(), blob_id);
        }
        for layer_id in 0..self.layer_names.len() {
            self.layer_names_index.insert(self.layer_names[layer_id].clone(), layer_id);
        }

        self.share_weights();
        self.debug_info = param.get_debug_info();
        if Caffe::root_solver() {
            info!("Network initialization done.");
        }
    }

    pub fn forward(&mut self, loss: &mut Option<T>) -> &BlobVec<T> {
        let v = self.forward_from_to(0, self.layers.len() - 1usize);
        if loss.is_some() {
            loss.replace(v);
        }

        &self.net_output_blobs
    }

    #[deprecated]
    pub fn forward_prefilled(&mut self, loss: &mut Option<T>) -> &BlobVec<T> {
        self.forward(loss)
    }

    pub fn forward_from_to(&mut self, start: usize, end: usize) -> T {
        check_ge!(start, 0);
        check_lt!(end, self.layers.len());

        let mut loss = T::default();
        for i in start..=end {
            for c in self.before_forward.iter_mut() {
                c.run(i);
            }
            let layer_loss = self.layers[i].borrow_mut().forward(&self.bottom_vecs[i], &self.top_vecs[i]);
            loss += layer_loss;

            if self.debug_info {
                self.forward_debug_info(i);
            }
            for c in self.after_forward.iter_mut() {
                c.run(i);
            }
        }

        loss
    }

    pub fn forward_from(&mut self, start: usize) -> T {
        self.forward_from_to(start, self.layers.len() - 1usize)
    }

    pub fn forward_to(&mut self, end: usize) -> T {
        self.forward_from_to(0, end)
    }

    #[deprecated]
    pub fn forward_with_input(&mut self, bottom: &BlobVec<T>, loss: &mut Option<T>) -> &BlobVec<T> {
        // Copy bottom to net bottoms
        for i in 0..bottom.len() {
            self.net_input_blobs[i].borrow_mut().copy_from(&*bottom[i].as_ref().borrow(), false, false);
        }

        self.forward(loss)
    }

    pub fn clear_param_diffs(&mut self) {
        for param in &self.learnable_params {
            let mut blob = param.borrow_mut();
            match Caffe::mode() {
                CaffeBrew::CPU => {
                    CaffeUtil::caffe_set(blob.count(), T::default(), blob.mutable_cpu_diff());
                }
                CaffeBrew::GPU => {
                    unimplemented!();
                }
            }
        }
    }

    pub fn backward(&mut self) {
        self.backward_from_to(self.layers.len() - 1usize, 0);
        if self.debug_info {
            let mut asum_data = T::default();
            let mut asum_diff = T::default();
            let mut sumsq_data = T::default();
            let mut sumsq_diff = T::default();
            for param in &self.learnable_params {
                let blob = param.as_ref().borrow();
                asum_data += blob.asum_data();
                asum_diff += blob.asum_diff();
                sumsq_data += blob.sumsq_data();
                sumsq_diff += blob.sumsq_diff();
            }

            let l2norm_data = T::sqrt(sumsq_data);
            let l2norm_diff = T::sqrt(sumsq_diff);
            error!("    [Backward] All net params (data, diff): L1 norm = ({:?}, {:?}); L2 norm = ({:?}, {:?})",
                asum_data, asum_diff, l2norm_data, l2norm_diff);
        }
    }

    pub fn backward_from_to(&mut self, start: usize, end: usize) {
        check_ge!(end, 0);
        check_lt!(start, self.layers.len());

        for i in (end..=start).rev() {
            for c in self.before_backward.iter_mut() {
                c.run(i);
            }
            if self.layer_need_backward[i] {
                self.layers[i].borrow_mut().backward(&self.top_vecs[i], &self.bottom_need_backward[i],
                                                     &self.bottom_vecs[i]);
                if self.debug_info {
                    self.backward_debug_info(i);
                }
            }
            for c in self.after_backward.iter_mut() {
                c.run(i);
            }
        }
    }

    pub fn backward_from(&mut self, start: usize) {
        self.backward_from_to(start, 0);
    }

    pub fn backward_to(&mut self, end: usize) {
        self.backward_from_to(self.layers.len() - 1usize, end);
    }

    pub fn reshape(&mut self) {
        for i in 0..self.layers.len() {
            self.layers[i].borrow_mut().reshape(&self.bottom_vecs[i], &self.top_vecs[i]);
        }
    }

    pub fn forward_backward(&mut self) -> T {
        let mut loss = Some(T::default());
        self.forward(&mut loss);
        self.backward();
        loss.unwrap()
    }

    pub fn update(&mut self) {
        for param in &self.learnable_params {
            param.borrow_mut().update();
        }
    }

    // todo: private usage.
    pub fn share_weights(&mut self) {
        for i in 0..self.params.len() {
            let idx = self.param_owners[i];
            if idx < 0 {
                continue;
            }

            let mut param = self.params[i].borrow_mut();
            let owner_param = self.params[idx as usize].as_ref().borrow();
            param.share_data(&*owner_param);
            param.share_diff(&*owner_param);
        }
    }

    pub fn share_trained_layers_with(&mut self, other: &Net<T>) {
        let num_source_layers = other.layers.len();
        for i in 0..num_source_layers {
            let source_layer_name = other.layer_names[i].as_str();
            let target_layer_id = self.layer_names.iter().position(|x| x == source_layer_name);
            if target_layer_id.is_none() {
                info!("Ignoring source layer {:?}", source_layer_name);
                continue;
            }

            info!("Copying source layer {:?}", source_layer_name);
            let target_layer_id = target_layer_id.unwrap();
            let source_blobs = other.layers[i].as_ref().borrow();
            let source_blobs = source_blobs.blobs();
            let target_blobs = self.layers[target_layer_id].as_ref().borrow();
            let target_blobs = target_blobs.blobs();
            check_eq!(target_blobs.len(), source_blobs.len(),
                "Incompatible number of blobs for layer {:?}", source_layer_name);
            for j in 0..target_blobs.len() {
                let source_blob = source_blobs[j].as_ref().borrow();
                {
                    let target_blob = target_blobs[j].as_ref().borrow();
                    assert_eq!(target_blob.shape(), source_blob.shape(),
                               "Cannot share param {:?} weights from layer '{:?}'; shape mismatch. Source param shape is \
                                {:?}; target param shape is {:?}",
                               j, source_layer_name, source_blob.shape_string(), target_blob.shape_string());
                }

                target_blobs[j].borrow_mut().share_data(&*source_blob);
            }
        }
    }

    pub fn copy_trained_layers_from(&mut self, param: &NetParameter) {
        for source_layer in param.get_layer() {
            let source_layer_name = source_layer.get_name();
            let target_layer_id = self.layer_names.iter().position(|x| x == source_layer_name);
            if target_layer_id.is_none() {
                info!("Ignoring source layer {:?}", source_layer_name);
                continue;
            }

            info!("Copying source layer {:?}", source_layer_name);
            let target_layer_id = target_layer_id.unwrap();
            let source_blobs = source_layer.get_blobs();
            let target_blobs = self.layers[target_layer_id].as_ref().borrow();
            let target_blobs = target_blobs.blobs();
            check_eq!(target_blobs.len(), source_blobs.len(),
                "Incompatible number of blobs for layer {:?}", source_layer_name);
            for j in 0..target_blobs.len() {
                if !target_blobs[j].as_ref().borrow().shape_equals(&source_blobs[j]) {
                    let mut source_blob = Blob::<T>::new();
                    source_blob.set_from_proto(&source_blobs[j], true);
                    assert!(false, "Cannot copy param {:?} weights from layer '{:?}'; shape mismatch. Source \
                        param shape is {:?}; target param shape is {:?}. To learn this layer's parameters from \
                        scratch rather than copying from a saved net, rename the layer.",
                        j, source_layer_name, source_blob.shape_string(),
                        target_blobs[j].as_ref().borrow().shape_string());
                }

                target_blobs[j].borrow_mut().set_from_proto(&source_blobs[j], false);
            }
        }
    }

    pub fn copy_trained_layers_from_file(&mut self, trained_filename: &str) {
        // todo: is hdf5 file
        self.copy_trained_layers_from_binary_proto(trained_filename);
    }

    pub fn copy_trained_layers_from_binary_proto(&mut self, trained_filename: &str) {
        // todo ReadNetParamsFromBinaryFileOrDie
        let mut param = NetParameter::new();

        self.copy_trained_layers_from(&param);
    }

    pub fn copy_trained_layers_from_hdf5(&mut self, _trained_filename: &str) {
        unimplemented!();
    }

    pub fn to_proto(&self, param: &mut NetParameter, write_diff: bool) {
        param.clear();
        param.set_name(Chars::from(self.name.as_str()));
        // Add bottom and top
        info!("Serializing {:?} layers", self.layers.len());
        for layer in &self.layers {
            layer.as_ref().borrow().to_proto(param.mut_layer().push_default(), write_diff);
        }
    }

    pub fn to_hdf5(&self, _filename: &str, _write_diff: bool) {
        unimplemented!();
    }

    /// Returns the network name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the layer names.
    #[inline]
    pub fn layer_names(&self) -> &Vec<String> {
        &self.layer_names
    }

    /// Returns the blob names.
    #[inline]
    pub fn blob_names(&self) -> &Vec<String> {
        &self.blob_names
    }

    /// Returns the blobs.
    #[inline]
    pub fn blobs(&self) -> &BlobVec<T> {
        &self.blobs
    }

    /// Return the layers.
    #[inline]
    pub fn layers(&self) -> &LayerVec<T> {
        &self.layers
    }

    /// Returns the phase: `TRAIN` or `TEST`.
    #[inline]
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Returns the bottom blob vectors for each layer.
    /// *usually you won't need this unless you do per-layer checks such as* **gradients**.
    #[inline]
    pub fn bottom_vecs(&self) -> &Vec<BlobVec<T>> {
        &self.bottom_vecs
    }

    /// Returns the top blob vectors for each layer.
    /// *usually you won't need this unless you do per-layer checks such as* **gradients**.
    #[inline]
    pub fn top_vecs(&self) -> &Vec<BlobVec<T>> {
        &self.top_vecs
    }

    /// Returns the ids of the top blobs of layer `i`.
    #[inline]
    pub fn top_ids(&self, i: usize) -> &Vec<usize> {
        check_lt!(i, self.top_id_vecs.len(), "Invalid layer id");

        &self.top_id_vecs[i]
    }

    /// Returns the ids of the bottom blobs of layer `i`.
    #[inline]
    pub fn bottom_ids(&self, i: usize) -> &Vec<usize> {
        check_lt!(i, self.bottom_id_vecs.len(), "Invalid layer id");

        &self.bottom_id_vecs[i]
    }

    #[inline]
    pub fn bottom_need_backward(&self) -> &Vec<Vec<bool>> {
        &self.bottom_need_backward
    }

    #[inline]
    pub fn blob_loss_weights(&self) -> &Vec<T> {
        &self.blob_loss_weights
    }

    #[inline]
    pub fn layer_need_backward(&self) -> &Vec<bool> {
        &self.layer_need_backward
    }

    /// Returns the parameters.
    #[inline]
    pub fn params(&self) -> &BlobVec<T> {
        &self.params
    }

    #[inline]
    pub fn learnable_params(&self) -> &BlobVec<T> {
        &self.learnable_params
    }

    /// Returns the learnable parameter learning rate multipliers.
    #[inline]
    pub fn params_lr(&self) -> &Vec<f32> {
        &self.params_lr
    }

    #[inline]
    pub fn has_params_lr(&self) -> &Vec<bool> {
        &self.has_params_lr
    }

    /// Returns the learnable parameter decay multipliers.
    #[inline]
    pub fn params_weight_decay(&self) -> &Vec<f32> {
        &self.params_weight_decay
    }

    #[inline]
    pub fn has_params_decay(&self) -> &Vec<bool> {
        &self.has_params_decay
    }

    #[inline]
    pub fn param_names_index(&self) -> &HashMap<String, usize> {
        &self.param_names_index
    }

    #[inline]
    pub fn param_owners(&self) -> &Vec<i32> {
        &self.param_owners
    }

    #[inline]
    pub fn param_display_names(&self) -> &Vec<String> {
        &self.param_display_names
    }

    /// Input blob numbers.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.net_input_blobs.len()
    }

    /// Output blob numbers
    #[inline]
    pub fn num_outputs(&self) -> usize {
        self.net_output_blobs.len()
    }

    #[inline]
    pub fn input_blobs(&self) -> &BlobVec<T> {
        &self.net_input_blobs
    }

    #[inline]
    pub fn output_blobs(&self) -> &BlobVec<T> {
        &self.net_output_blobs
    }

    #[inline]
    pub fn input_blob_indices(&self) -> &Vec<usize> {
        &self.net_input_blob_indices
    }

    #[inline]
    pub fn output_blob_indices(&self) -> &Vec<usize> {
        &self.net_output_blob_indices
    }

    #[inline]
    pub fn has_blob(&self, blob_name: &str) -> bool {
        self.blob_names_index.contains_key(blob_name)
    }

    pub fn blob_by_name(&self, blob_name: &str) -> Option<SharedBlob<T>> {
        let idx = self.blob_names_index.get(blob_name);
        idx.map_or_else(|| {
            warn!("Unknown blob name {:?}", blob_name);
            Option::None
        }, |&i| Some(self.blobs[i].clone()))
    }

    #[inline]
    pub fn has_layer(&self, layer_name: &str) -> bool {
        self.layer_names_index.contains_key(layer_name)
    }

    pub fn layer_by_name(&self, layer_name: &str) -> Option<SharedLayer<T>> {
        self.layer_names_index.get(layer_name).map_or_else(|| {
            warn!("Unknown layer name {:?}", layer_name);
            Option::None
        }, |&i| Some(self.layers[i].clone()))
    }

    #[inline]
    pub fn set_debug_info(&mut self, value: bool) {
        self.debug_info = value;
    }

    // helpers for init.
    pub fn filter_net(param: &NetParameter, param_filtered: &mut NetParameter) {
        let net_state = param.get_state();
        param_filtered.clone_from(param);
        param_filtered.clear_layer();

        for layer_param in param.get_layer() {
            let layer_name = layer_param.get_name();
            assert!(layer_param.get_include().is_empty() || layer_param.get_exclude().is_empty(),
                    "Specify either include rules or exclude rules; not both.");
            // If no include rules are specified, the layer is included by default and only
            // excluded if it meets one of the exclude rules.
            let mut layer_included = layer_param.get_include().is_empty();
            let exclude = layer_param.get_exclude();
            let mut j = 0usize;
            while layer_included && j < exclude.len() {
                if Self::state_meets_rule(net_state, &exclude[j], layer_name) {
                    layer_included = false;
                }
                j += 1;
            }

            j = 0;
            let include = layer_param.get_include();
            while !layer_included && j < include.len() {
                if Self::state_meets_rule(net_state, &include[j], layer_name) {
                    layer_included = true;
                }
                j += 1;
            }

            if layer_included {
                param_filtered.mut_layer().push_default().clone_from(layer_param);
            }
        }
    }

    pub fn state_meets_rule(state: &NetState, rule: &NetStateRule, layer_name: &str) -> bool {
        // Check whether the rule is broken due to phase.
        if rule.has_phase() {
            if rule.get_phase() != state.get_phase() {
                if Caffe::root_solver() {
                    info!("The NetState phase ({:?}) differed from the phase ({:?}) specified by a rule in layer {:?}",
                          state.get_phase(), rule.get_phase(), layer_name);
                }
                return false;
            }
        }
        // Check whether the rule is broken due to min level.
        if rule.has_min_level() {
            if state.get_level() < rule.get_min_level() {
                if Caffe::root_solver() {
                    info!("The NetState level ({:?}) is below the min_level ({:?}) specified by a rule in layer {:?}",
                          state.get_level(), rule.get_min_level(), layer_name);
                }
                return false;
            }
        }
        // Check whether the rule is broken due to max level.
        if rule.has_max_level() {
            if state.get_level() > rule.get_max_level() {
                if Caffe::root_solver() {
                    info!("The NetState level ({:?}) is above the max_level ({:?}) specified by a rule in layer {:?}",
                          state.get_level(), rule.get_max_level(), layer_name);
                }
                return false;
            }
        }
        // Check whether the rule is broken due to stage. The NetState must contain ALL
        // of the rule's stages to meet it.
        for rule_stage in rule.get_stage() {
            // Check that the NetState contains the rule's i-th stage.
            let mut has_stage = false;
            for state_stage in state.get_stage() {
                if rule_stage == state_stage {
                    has_stage = true;
                    break;
                }
            }
            if !has_stage {
                if Caffe::root_solver() {
                    info!("The NetState did not contain stage '{:?}' specified by a rule in layer {:?}",
                          rule_stage, layer_name);
                }
                return false;
            }
        }
        // Check whether the rule is broken due to not_stage. The NetStage must contain
        // NONE of the rule's not_stages to meet it.
        for rule_stage in rule.get_not_stage() {
            // Check that the NetState contains the rule's i-th not_stage.
            let mut has_stage = false;
            for state_stage in state.get_stage() {
                if rule_stage == state_stage {
                    has_stage = true;
                    break;
                }
            }
            if has_stage {
                if Caffe::root_solver() {
                    info!("The NetState contained a not_stage '{:?}' specified by a rule in layer {:?}",
                          rule_stage, layer_name);
                }
                return false;
            }
        }

        true
    }

    // Invoked callbacks.
    pub fn before_forward(&self) -> &Vec<Box<dyn NetCallback>> {
        &self.before_forward
    }

    pub fn add_before_forward(&mut self, value: Box<dyn NetCallback>) {
        self.before_forward.push(value);
    }

    pub fn after_forward(&self) -> &Vec<Box<dyn NetCallback>> {
        &self.after_forward
    }

    pub fn add_after_forward(&mut self, value: Box<dyn NetCallback>) {
        self.after_forward.push(value);
    }

    pub fn before_backward(&self) -> &Vec<Box<dyn NetCallback>> {
        &self.before_backward
    }

    pub fn add_before_backward(&mut self, value: Box<dyn NetCallback>) {
        self.before_backward.push(value);
    }

    pub fn after_backward(&self) -> &Vec<Box<dyn NetCallback>> {
        &self.after_backward
    }

    pub fn add_after_backward(&mut self, value: Box<dyn NetCallback>) {
        self.after_backward.push(value);
    }

    // protected
    // helpers for init.
    /// Append a net top blob to the net.
    fn append_top(&mut self, param: &NetParameter, layer_id: usize, top_id: usize,
                  available_blobs: &mut Option<HashSet<String>>,
                  blob_name_to_idx: &mut Option<HashMap<String, usize>>) {
        let layer_param = &param.get_layer()[layer_id];
        let blob_name = if layer_param.get_top().len() > top_id {
            layer_param.get_top()[top_id].as_ref()
        } else {
            "(automatic)"
        };

        // Check if we are doing in-place computation
        if blob_name_to_idx.is_some() &&
            layer_param.get_bottom().len() > top_id &&
            blob_name == &*layer_param.get_bottom()[top_id] {
            let blob_name_to_idx = blob_name_to_idx.as_ref().unwrap();
            // In-place computation
            if Caffe::root_solver() {
                info!("{:?} -> {:?} (in-place)", layer_param.get_name(), blob_name);
            }
            let idx = blob_name_to_idx[blob_name];
            self.top_vecs[layer_id].push(self.blobs[idx].clone());
            self.top_id_vecs[layer_id].push(idx);
        } else if blob_name_to_idx.as_ref().map_or(false, |b| b.contains_key(blob_name)) {
            // If we are not doing in-place computation but have duplicated blobs,
            // raise an error.
            assert!(false, "Top blob '{:?}' produced by multiple sources.", blob_name);
        } else {
            // Normal output.
            if Caffe::root_solver() {
                info!("{:?} -> {:?}", layer_param.get_name(), blob_name);
            }
            let blob_pointer = SharedBlob::new(RefCell::new(Blob::new()));
            let blob_id = self.blobs.len();
            self.blobs.push(blob_pointer.clone());
            self.blob_names.push(blob_name.to_string());
            self.blob_need_backward.push(false);
            if let &mut Some(ref mut blob_name_to_idx) = blob_name_to_idx {
                blob_name_to_idx.insert(blob_name.to_string(), blob_id);
            }
            self.top_id_vecs[layer_id].push(blob_id);
            self.top_vecs[layer_id].push(blob_pointer);
        }

        if let &mut Some(ref mut available_blobs) = available_blobs {
            available_blobs.insert(blob_name.to_string());
        }
    }

    /// Append a net bottom blob to the net.
    fn append_bottom(&mut self, param: &NetParameter, layer_id: usize, bottom_id: usize,
                     available_blobs: &mut HashSet<String>,
                     blob_name_to_idx: &mut HashMap<String, usize>) -> usize {
        let layer_param = &param.get_layer()[layer_id];
        let blob_name: &str = layer_param.get_bottom()[bottom_id].as_ref();
        if !available_blobs.contains(blob_name) {
            assert!(false, "Unknown bottom blob '{:?}' (layer '{:?}', bottom index {:?})",
                    blob_name, layer_param.get_name(), bottom_id);
        }

        let blob_id = blob_name_to_idx[blob_name];
        if Caffe::root_solver() {
            info!("{:?} <- {:?}", self.layer_names[layer_id], blob_name);
        }
        self.bottom_vecs[layer_id].push(self.blobs[blob_id].clone());
        self.bottom_id_vecs[layer_id].push(blob_id);
        available_blobs.remove(blob_name);
        let mut need_backward = self.blob_need_backward[blob_id];
        // Check if the backpropagation on bottom_id should be skipped
        if !layer_param.get_propagate_down().is_empty() {
            need_backward = layer_param.get_propagate_down()[bottom_id];
        }
        self.bottom_need_backward[layer_id].push(need_backward);

        blob_id
    }

    /// Append a net parameter blob to the net.
    fn append_param(&mut self, param: &NetParameter, layer_id: usize, param_id: usize) {
        let layer = self.layers[layer_id].as_ref().borrow();
        let layer_param = layer.layer_param();
        let param_size = layer_param.get_param().len();
        let param_name = if param_size > param_id {
            layer_param.get_param()[param_id].get_name()
        } else {
            ""
        };

        if param_name.is_empty() {
            self.param_display_names.push(param_id.to_string());
        } else {
            self.param_display_names.push(param_name.to_string());
        }

        let net_param_id = self.params.len();
        self.params.push(layer.blobs()[param_id].clone());
        self.param_id_vecs[layer_id].push(net_param_id);
        self.param_layer_indices.push((layer_id, param_id));

        let default_param_spec = ParamSpec::new();
        let param_spec = if param_size > param_id {
            &layer_param.get_param()[param_id]
        } else {
            &default_param_spec
        };
        if param_size == 0 || param_name.is_empty() || !self.param_names_index.contains_key(param_name) {
            // This layer "owns" this parameter blob -- it is either anonymous (i.e., not given
            // a param_name) or explicitly given a name that we haven't already seen.
            self.param_owners.push(-1);
            if !param_name.is_empty() {
                self.param_names_index.insert(param_name.to_string(), net_param_id);
            }
            let learnable_param_id = self.learnable_params.len();
            self.learnable_params.push(self.params[net_param_id].clone());
            self.learnable_param_ids.push(learnable_param_id);
            self.has_params_lr.push(param_spec.has_lr_mult());
            self.has_params_decay.push(param_spec.has_decay_mult());
            self.params_lr.push(param_spec.get_lr_mult());
            self.params_weight_decay.push(param_spec.get_decay_mult());
        } else {
            // Named param blob with name we've seen before: share params
            let owner_net_param_id = self.param_names_index[param_name];
            self.param_owners.push(owner_net_param_id as i32);
            let (owner_layer_id, owner_param_id) = self.param_layer_indices[owner_net_param_id];
            if Caffe::root_solver() {
                info!("Sharing parameters '{:?}' owned by layer '{:?}', param index {:?}",
                      param_name, self.layer_names[owner_layer_id], owner_param_id);
            }

            let this_blob = layer.blobs()[param_id].as_ref().borrow();
            let owner_blob = self.layers[owner_layer_id].as_ref().borrow();
            let owner_blob = owner_blob.blobs()[owner_param_id].as_ref().borrow();
            if param_size > param_id &&
                layer_param.get_param()[param_id].get_share_mode() == ParamSpec_DimCheckMode::PERMISSIVE {
                // Permissive dimension checking -- only check counts are the same.
                check_eq!(this_blob.count(), owner_blob.count(),
                    "Cannot share param '{:?}' owned by layer '{:?}' with layer '{:?}'; count mismatch. \
                    Owner layer param shape is {:?}; sharing layer shape is {:?}",
                    param_name, self.layer_names[owner_layer_id], self.layer_names[layer_id],
                    owner_blob.shape_string(), this_blob.shape_string());
            } else {
                // Strict dimension checking -- all dims must be the same.
                check_eq!(this_blob.shape(), owner_blob.shape(),
                    "Cannot share param '{:?}' owned by layer '{:?}' with layer '{:?}'; count mismatch. \
                    Owner layer param shape is {:?}; sharing layer shape is {:?}",
                    param_name, self.layer_names[owner_layer_id], self.layer_names[layer_id],
                    owner_blob.shape_string(), this_blob.shape_string());
            }

            let learnable_param_id = self.learnable_param_ids[owner_net_param_id];
            self.learnable_param_ids.push(learnable_param_id);
            if param_spec.has_lr_mult() {
                if self.has_params_lr[learnable_param_id] {
                    check_eq!(param_spec.get_lr_mult(), self.params_lr[learnable_param_id],
                        "Shared param '{:?}' has mismatched lr_mult.", param_name);
                } else {
                    self.has_params_lr[learnable_param_id] = true;
                    self.params_lr[learnable_param_id] = param_spec.get_lr_mult();
                }
            }
            if param_spec.has_decay_mult() {
                if self.has_params_decay[learnable_param_id] {
                    check_eq!(param_spec.get_decay_mult(), self.params_weight_decay[learnable_param_id],
                        "Shared param '{:?}' has mismatched decay_mult.", param_name);
                } else {
                    self.has_params_decay[learnable_param_id] = true;
                    self.params_weight_decay[learnable_param_id] = param_spec.get_decay_mult();
                }
            }
        }
    }

    /// Helper for displaying debug info in `forward`.
    fn forward_debug_info(&self, layer_id: usize) {
        for top_id in 0..self.top_vecs[layer_id].len() {
            let blob = self.top_vecs[layer_id][top_id].as_ref().borrow();
            let blob_name = self.blob_names[self.top_id_vecs[layer_id][top_id]].as_str();
            let data_abs_val_mean = blob.asum_data() / T::from_usize(blob.count());
            let data_abs_val_mean = T::from_div(data_abs_val_mean);
            if Caffe::root_solver() {
                info!("    [Forward] Layer {:?}, top blob {:?} data: {:?}",
                    self.layer_names[layer_id], blob_name, data_abs_val_mean);
            }
        }
        for param_id in 0..self.layers[layer_id].as_ref().borrow().blobs().len() {
            let layer = self.layers[layer_id].as_ref().borrow();
            let blob = layer.blobs()[param_id].as_ref().borrow();
            let net_param_id = self.param_id_vecs[layer_id][param_id];
            let blob_name = self.param_display_names[net_param_id].as_str();
            let data_abs_val_mean = blob.asum_data() / T::from_usize(blob.count());
            let data_abs_val_mean = T::from_div(data_abs_val_mean);
            if Caffe::root_solver() {
                info!("    [Forward] Layer {:?}, param blob {:?} data: {:?}",
                    self.layer_names[layer_id], blob_name, data_abs_val_mean);
            }
        }
    }

    /// Helper for displaying debug info in `backward`.
    fn backward_debug_info(&self, layer_id: usize) {
        let bottom_vec = &self.bottom_vecs[layer_id];
        for bottom_id in 0..bottom_vec.len() {
            if !self.bottom_need_backward[layer_id][bottom_id] {
                continue;
            }

            let blob = bottom_vec[bottom_id].as_ref().borrow();
            let blob_name = self.blob_names[self.bottom_id_vecs[layer_id][bottom_id]].as_str();
            let diff_abs_val_mean = blob.asum_diff() / T::from_usize(blob.count());
            let diff_abs_val_mean = T::from_div(diff_abs_val_mean);
            if Caffe::root_solver() {
                info!("    [Backward] Layer {:?}, bottom blob {:?} diff: {:?}",
                    self.layer_names[layer_id], blob_name, diff_abs_val_mean);
            }
        }
        for param_id in 0..self.layers[layer_id].as_ref().borrow().blobs().len() {
            if !self.layers[layer_id].as_ref().borrow().param_propagate_down(param_id) {
                continue;
            }

            let layer = self.layers[layer_id].as_ref().borrow();
            let blob = layer.blobs()[param_id].as_ref().borrow();
            let diff_abs_val_mean = blob.asum_diff() / T::from_usize(blob.count());
            let diff_abs_val_mean = T::from_div(diff_abs_val_mean);
            if Caffe::root_solver() {
                info!("    [Backward] Layer {:?}, param blob {:?} diff: {:?}",
                    self.layer_names[layer_id], param_id, diff_abs_val_mean);
            }
        }
    }

    /// Helper for displaying debug info in `update`.
    fn update_debug_info(&self, param_id: usize) {
        let blob = self.params[param_id].as_ref().borrow();
        let param_owner = self.param_owners[param_id];
        let layer_name = self.layer_names[self.param_layer_indices[param_id].0].as_str();
        let param_display_name = self.param_display_names[param_id].as_str();
        let diff_abs_val_mean = blob.asum_diff() / T::from_usize(blob.count());
        let diff_abs_val_mean = T::from_div(diff_abs_val_mean);
        if param_owner < 0 {
            let data_abs_val_mean = blob.asum_data() / T::from_usize(blob.count());
            let data_abs_val_mean = T::from_div(data_abs_val_mean);
            if Caffe::root_solver() {
                info!("    [Update] Layer {:?}, param {:?} data: {:?}, diff: {:?}",
                    layer_name, param_display_name, data_abs_val_mean, diff_abs_val_mean);
            }
        } else {
            let owner_layer_name = self.layer_names[self.param_layer_indices[param_owner as usize].0].as_str();
            let owner_display_name = self.param_display_names[param_owner as usize].as_str();
            if Caffe::root_solver() {
                info!("    [Update] Layer {:?}, param blob {:?} (owned by layer {:?}, param {:?}) diff: {:?}",
                    layer_name, param_display_name, owner_layer_name, owner_display_name, diff_abs_val_mean);
            }
        }
    }
}
