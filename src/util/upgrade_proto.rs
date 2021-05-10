use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::proto::caffe::{
    NetParameter, V1LayerParameter, V1LayerParameter_LayerType, LayerParameter, SolverParameter,
    V0LayerParameter_PoolMethod, PoolingParameter_PoolMethod, V1LayerParameter_DimCheckMode,
    ParamSpec_DimCheckMode, SolverParameter_SolverType
};
use crate::util::io::{read_proto_from_text_file, read_proto_from_binary_file};
use protobuf::{Clear, Chars};


/// Return true if the net is not the **current version**.
pub fn net_needs_upgrade(net_param: &NetParameter) -> bool {
    net_needs_v0_to_v1_upgrade(net_param) || net_needs_v1_to_v2_upgrade(net_param) ||
        net_needs_data_upgrade(net_param) || net_needs_input_upgrade(net_param) ||
        net_needs_batch_norm_upgrade(net_param)
}

/// Check for deprecations and upgrade the `NetParameter` as needed.
pub fn upgrade_net_as_needed(param_file: &str, param: &mut NetParameter) -> bool {
    let mut success = true;
    if net_needs_v0_to_v1_upgrade(param) {
        // NetParameter was specified using the old style (V0LayerParameter); try to upgrade it.
        info!("Attempting to upgrade input file specified using deprecated V0LayerParameter: {:?}", param_file);
        let original_param = param.clone();
        if !upgrade_v0_net(&original_param, param) {
            success = false;
            error!("Warning: had one or more problems upgrading V0NetParameter to NetParameter \
                (see above); continuing anyway.");
        } else {
            info!("Successfully upgraded file specified using deprecated V0LayerParameter.");
        }
        warn!("Note that future Caffe releases will not support V0NetParameter; using \
            ./build/tools/upgrade_net_proto_text for prototxt and ./build/tools/upgrade_net_proto_binary \
            for model weights upgrade this and any other net proto(s) to the new format.");
    }
    // NetParameter uses old style data transformation fields; try to upgrade it.
    if net_needs_data_upgrade(param) {
        info!("Attempting to upgrade input file specified using deprecated transformation params: {:?}", param_file);
        upgrade_net_data_transformation(param);
        info!("Successfully upgraded file specified using deprecated data transformation parameters.");
        warn!("Note that future Caffe releases will only support transform_param messages for transformation fields.");
    }
    if net_needs_v1_to_v2_upgrade(param) {
        info!("Attempting to upgrade input file specified using deprecated V1LayerParameter: {:?}", param_file);
        let original_param = param.clone();
        if !upgrade_v1_net(&original_param, param) {
            success = false;
            error!("Warning: had one or more problems upgrading V1LayerParameter (see above); continuing anyway.");
        } else {
            info!("Successfully upgraded file specified using deprecated V1LayerParameter");
        }
    }
    // NetParameter uses old style input fields; try to upgrade it.
    if net_needs_input_upgrade(param) {
        info!("Attempting to upgrade input file specified using deprecated input fields: {:?}", param_file);
        upgrade_net_input(param);
        info!("Successfully upgraded file specified using deprecated input fields.");
        warn!("Note that future Caffe releases will only support input layers and not input fields.");
    }
    // NetParameter uses old style batch norm layers; try to upgrade it.
    if net_needs_batch_norm_upgrade(param) {
        info!("Attempting to upgrade batch norm layers using deprecated params: {:?}", param_file);
        upgrade_net_batch_norm(param);
        info!("Successfully upgraded batch norm layers using deprecated params.");
    }

    success
}

/// Read parameters from a file into a `NetParameter` proto message.
pub fn read_net_params_from_text_file_or_die(param_file: &str, param: &mut NetParameter) {
    let c = read_proto_from_text_file(param_file, param);
    assert!(c, "Failed to parse NetParameter file: {:?}", param_file);
    upgrade_net_as_needed(param_file, param);
}

/// Read parameters from a file into a `NetParameter` proto message.
pub fn read_net_params_from_binary_file_or_die(param_file: &str, param: &mut NetParameter) {
    let c = read_proto_from_binary_file(param_file, param);
    assert!(c, "Failed to parse NetParameter file: {:?}", param_file);
    upgrade_net_as_needed(param_file, param);
}

/// Return true if any layer contains parameters specified using *deprecated* `V0LayerParameter`.
pub fn net_needs_v0_to_v1_upgrade(net_param: &NetParameter) -> bool {
    for layer in net_param.get_layers() {
        if layer.has_layer() {
            return true;
        }
    }

    false
}

/// Perform all necessary transformations to upgrade a `V0NetParameter` into a
/// `NetParameter` (including upgrading padding layers and `LayerParameter`).
pub fn upgrade_v0_net(v0_net_param_padding_layers: &NetParameter, net_param: &mut NetParameter) -> bool {
    // First upgrade padding layers to padded conv layers.
    let mut v0_net_param = NetParameter::new();
    upgrade_v0_padding_layers(v0_net_param_padding_layers, &mut v0_net_param);
    // Now upgrade layer parameters.
    let mut is_fully_compatible = true;
    net_param.clear();
    if v0_net_param.has_name() {
        net_param.set_name(Chars::from(v0_net_param.get_name()));
    }
    for layer in v0_net_param.get_layers() {
        is_fully_compatible &= upgrade_v0_layer_parameter(layer, net_param.mut_layers().push_default());
    }
    for input in v0_net_param.get_input() {
        net_param.mut_input().push(input.clone());
    }
    for input_dim in v0_net_param.get_input_dim() {
        net_param.mut_input_dim().push(*input_dim);
    }
    if v0_net_param.has_force_backward() {
        net_param.set_force_backward(v0_net_param.get_force_backward());
    }

    is_fully_compatible
}

/// Upgrade `NetParameter` with padding layers to pad-aware conv layers. For any padding
/// layer, remove it and put its pad parameter in any layers taking its top blob as input.
/// Error if any of these above layers are not-conv layers.
pub fn upgrade_v0_padding_layers(param: &NetParameter, param_upgraded_pad: &mut NetParameter) {
    // Copy everything other than the layers from the original param.
    param_upgraded_pad.clear();
    param_upgraded_pad.clone_from(param);
    param_upgraded_pad.clear_layers();
    // Figure out which layer each bottom blob comes from.
    let mut blob_name_to_last_top_idx = HashMap::new();
    for input in param.get_input() {
        blob_name_to_last_top_idx.insert(input.to_string(), -1);
    }

    let mut i = 0;
    for layer_connection in param.get_layers() {
        let layer_param = layer_connection.get_layer();
        // Add the layer to the new net, unless it's a padding layer.
        if layer_param.get_field_type() != "padding" {
            param_upgraded_pad.mut_layers().push_default().clone_from(layer_connection);
        }

        let mut j = 0;
        for blob_name in layer_connection.get_bottom() {
            let blob_name: &str = blob_name.as_ref();
            if !blob_name_to_last_top_idx.contains_key(blob_name) {
                assert!(false, "Unknown blob input {:?} to layer {:?}", blob_name, j);
            }

            let top_idx = blob_name_to_last_top_idx[blob_name];
            if top_idx == -1 {
                j += 1;
                continue;
            }

            let source_layer = &param.get_layers()[top_idx as usize];
            if source_layer.get_layer().get_field_type() == "padding" {
                // This layer has a padding layer as input -- check that it is a conv layer or a
                // pooling layer and takes only one input. Also check that the padding layer input
                // has only one input and one output. Other cases have undefined behavior in Caffe.
                assert!((layer_param.get_field_type() == "conv") || (layer_param.get_field_type() == "pool"),
                        "Padding layer input to non-convolutional / non-pooling layer type {:?}",
                        layer_param.get_field_type());
                check_eq!(layer_connection.get_bottom().len(), 1, "Conv Layer takes a single blob as input.");
                check_eq!(source_layer.get_bottom().len(), 1, "Padding Layer takes a single blob as input.");
                check_eq!(source_layer.get_top().len(), 1, "Padding Layer produces a single blob as output.");

                let layer_index = param_upgraded_pad.get_layers().len() - 1usize;
                param_upgraded_pad.mut_layers()[layer_index].mut_layer().set_pad(source_layer.get_layer().get_pad());
                param_upgraded_pad.mut_layers()[layer_index].mut_bottom()[j as usize] = source_layer.get_bottom()[0].clone();
            }

            j += 1;
        }

        for blob_name in layer_connection.get_top() {
            blob_name_to_last_top_idx.insert(blob_name.to_string(), i);
        }

        i += 1;
    }
}

/// Upgrade a single `V0LayerConnection` to the `V1LayerParameter` format.
pub fn upgrade_v0_layer_parameter(v0_layer_connection: &V1LayerParameter, layer_param: &mut V1LayerParameter) -> bool {
    let mut is_fully_compatible = true;
    layer_param.clear();
    for b in v0_layer_connection.get_bottom() {
        layer_param.mut_bottom().push(b.clone());
    }
    for t in v0_layer_connection.get_top() {
        layer_param.mut_top().push(t.clone());
    }
    if !v0_layer_connection.has_layer() {
        return is_fully_compatible;
    }

    let v0_layer_param = v0_layer_connection.get_layer();
    if v0_layer_param.has_name() {
        layer_param.set_name(Chars::from(v0_layer_param.get_name()));
    }
    let ty = v0_layer_param.get_field_type();
    if v0_layer_param.has_field_type() {
        layer_param.set_field_type(upgrade_v0_layer_type(ty));
    }
    for blob in v0_layer_param.get_blobs() {
        layer_param.mut_blobs().push_default().clone_from(blob);
    }
    for blob_lr in v0_layer_param.get_blobs_lr() {
        layer_param.mut_blobs_lr().push(*blob_lr);
    }
    for decay in v0_layer_param.get_weight_decay() {
        layer_param.mut_weight_decay().push(*decay);
    }
    if v0_layer_param.has_num_output() {
        if ty == "conv" {
            layer_param.mut_convolution_param().set_num_output(v0_layer_param.get_num_output());
        } else if ty == "innerproduce" {
            layer_param.mut_inner_product_param().set_num_output(v0_layer_param.get_num_output());
        } else {
            error!("Unknown parameter num_output for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_biasterm() {
        if ty == "conv" {
            layer_param.mut_convolution_param().set_bias_term(v0_layer_param.get_biasterm());
        } else if ty == "innerproduct" {
            layer_param.mut_inner_product_param().set_bias_term(v0_layer_param.get_biasterm());
        } else {
            error!("Unknown parameter biasterm for layer type: {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_weight_filler() {
        if ty == "conv" {
            layer_param.mut_convolution_param().mut_weight_filler().clone_from(v0_layer_param.get_weight_filler());
        } else if ty == "innerproduct" {
            layer_param.mut_inner_product_param().mut_weight_filler().clone_from(v0_layer_param.get_weight_filler());
        } else {
            error!("Unknown parameter weight_filler for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_bias_filler() {
        if ty == "conv" {
            layer_param.mut_convolution_param().mut_bias_filler().clone_from(v0_layer_param.get_bias_filler());
        } else if ty == "innerproduct" {
            layer_param.mut_inner_product_param().mut_bias_filler().clone_from(v0_layer_param.get_bias_filler());
        } else {
            error!("Unknown parameter bias_filler for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_pad() {
        if ty == "conv" {
            layer_param.mut_convolution_param().mut_pad().push(v0_layer_param.get_pad());
        } else if ty == "pool" {
            layer_param.mut_pooling_param().set_pad(v0_layer_param.get_pad());
        } else {
            error!("Unknown parameter pad for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_kernelsize() {
        if ty == "conv" {
            layer_param.mut_convolution_param().mut_kernel_size().push(v0_layer_param.get_kernelsize());
        } else if ty == "pool" {
            layer_param.mut_pooling_param().set_kernel_size(v0_layer_param.get_kernelsize());
        } else {
            error!("Unknown parameter kernelsize for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_group() {
        if ty == "conv" {
            layer_param.mut_convolution_param().set_group(v0_layer_param.get_group());
        } else {
            error!("Unknown parameter group for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_stride() {
        if ty == "conv" {
            layer_param.mut_convolution_param().mut_stride().push(v0_layer_param.get_stride());
        } else if ty == "pool" {
            layer_param.mut_pooling_param().set_stride(v0_layer_param.get_stride());
        } else {
            error!("Unknown parameter stride for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_pool() {
        if ty == "pool" {
            let pool = v0_layer_param.get_pool();
            match pool {
                V0LayerParameter_PoolMethod::MAX => {
                    layer_param.mut_pooling_param().set_pool(PoolingParameter_PoolMethod::MAX);
                }
                V0LayerParameter_PoolMethod::AVE => {
                    layer_param.mut_pooling_param().set_pool(PoolingParameter_PoolMethod::AVE);
                }
                V0LayerParameter_PoolMethod::STOCHASTIC => {
                    layer_param.mut_pooling_param().set_pool(PoolingParameter_PoolMethod::STOCHASTIC);
                }
            }
            // other match: unknown pool method.
        } else {
            error!("Unknown parameter pool for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_dropout_ratio() {
        if ty == "dropout" {
            layer_param.mut_dropout_param().set_dropout_ratio(v0_layer_param.get_dropout_ratio());
        } else {
            error!("Unknown parameter dropout_ratio for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_local_size() {
        if ty == "lrn" {
            layer_param.mut_lrn_param().set_local_size(v0_layer_param.get_local_size());
        } else {
            error!("Unknown parameter local_size for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_alpha() {
        if ty == "lrn" {
            layer_param.mut_lrn_param().set_alpha(v0_layer_param.get_alpha());
        } else {
            error!("Unknown parameter alpha for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_beta() {
        if ty == "lrn" {
            layer_param.mut_lrn_param().set_beta(v0_layer_param.get_beta());
        } else {
            error!("Unknown parameter beta for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_k() {
        if ty == "lrn" {
            layer_param.mut_lrn_param().set_k(v0_layer_param.get_k());
        } else {
            error!("Unknown parameter k for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_source() {
        if ty == "data" {
            layer_param.mut_data_param().set_source(Chars::from(v0_layer_param.get_source()));
        } else if ty == "hdf5_data" {
            layer_param.mut_hdf5_data_param().set_source(Chars::from(v0_layer_param.get_source()));
        } else if ty == "images" {
            layer_param.mut_image_data_param().set_source(Chars::from(v0_layer_param.get_source()));
        } else if ty == "window_data" {
            layer_param.mut_window_data_param().set_source(Chars::from(v0_layer_param.get_source()));
        } else if ty == "infogain_loss" {
            layer_param.mut_infogain_loss_param().set_source(Chars::from(v0_layer_param.get_source()));
        } else {
            error!("Unknown parameter source for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_scale() {
        layer_param.mut_transform_param().set_scale(v0_layer_param.get_scale());
    }
    if v0_layer_param.has_meanfile() {
        layer_param.mut_transform_param().set_mean_file(Chars::from(v0_layer_param.get_meanfile()));
    }
    if v0_layer_param.has_batchsize() {
        if ty == "data" {
            layer_param.mut_data_param().set_batch_size(v0_layer_param.get_batchsize());
        } else if ty == "hdf5_data" {
            layer_param.mut_hdf5_data_param().set_batch_size(v0_layer_param.get_batchsize());
        } else if ty == "images" {
            layer_param.mut_image_data_param().set_batch_size(v0_layer_param.get_batchsize());
        } else if ty == "window_data" {
            layer_param.mut_window_data_param().set_batch_size(v0_layer_param.get_batchsize());
        } else {
            error!("Unknown parameter batchsize for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_cropsize() {
        layer_param.mut_transform_param().set_crop_size(v0_layer_param.get_cropsize());
    }
    if v0_layer_param.has_mirror() {
        layer_param.mut_transform_param().set_mirror(v0_layer_param.get_mirror());
    }
    if v0_layer_param.has_rand_skip() {
        if ty == "data" {
            layer_param.mut_data_param().set_rand_skip(v0_layer_param.get_rand_skip());
        } else if ty == "images" {
            layer_param.mut_image_data_param().set_rand_skip(v0_layer_param.get_rand_skip());
        } else {
            error!("Unknown parameter rand_skip for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_shuffle_images() {
        if ty == "images" {
            layer_param.mut_image_data_param().set_shuffle(v0_layer_param.get_shuffle_images());
        } else {
            error!("Unknown parameter shuffle for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_new_height() {
        if ty == "images" {
            layer_param.mut_image_data_param().set_new_height(v0_layer_param.get_new_height() as u32);
        } else {
            error!("Unknown parameter new_height for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_new_width() {
        if ty == "images" {
            layer_param.mut_image_data_param().set_new_width(v0_layer_param.get_new_width() as u32);
        } else {
            error!("Unknown parameter new_width for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_concat_dim() {
        if ty == "concat" {
            layer_param.mut_concat_param().set_concat_dim(v0_layer_param.get_concat_dim());
        } else {
            error!("Unknown parameter concat_dim for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_det_fg_threshold() {
        if ty == "window_data" {
            layer_param.mut_window_data_param().set_fg_threshold(v0_layer_param.get_det_fg_threshold());
        } else {
            error!("Unknown parameter det_fg_threshold for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_det_bg_threshold() {
        if ty == "window_data" {
            layer_param.mut_window_data_param().set_bg_threshold(v0_layer_param.get_det_bg_threshold());
        } else {
            error!("Unknown parameter det_bg_threshold for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_det_fg_fraction() {
        if ty == "window_data" {
            layer_param.mut_window_data_param().set_fg_fraction(v0_layer_param.get_det_fg_fraction());
        } else {
            error!("Unknown parameter det_ft_fraction for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_det_context_pad() {
        if ty == "window_data" {
            layer_param.mut_window_data_param().set_context_pad(v0_layer_param.get_det_context_pad());
        } else {
            error!("Unknown parameter det_context_pad for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_det_crop_mode() {
        if ty == "window_data" {
            layer_param.mut_window_data_param().set_crop_mode(Chars::from(v0_layer_param.get_det_crop_mode()));
        } else {
            error!("Unknown parameter det_crop_mode for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }
    if v0_layer_param.has_hdf5_output_param() {
        if ty == "hdf5_output" {
            layer_param.mut_hdf5_output_param().clone_from(v0_layer_param.get_hdf5_output_param());
        } else {
            error!("Unknown parameter hdf5_output_param for layer type {:?}", ty);
            is_fully_compatible = false;
        }
    }

    is_fully_compatible
}

pub fn upgrade_v0_layer_type(ty: &str) -> V1LayerParameter_LayerType {
    match ty {
        "accuracy" => V1LayerParameter_LayerType::ACCURACY,
        "bnll" => V1LayerParameter_LayerType::BNLL,
        "concat" => V1LayerParameter_LayerType::CONCAT,
        "conv" => V1LayerParameter_LayerType::CONVOLUTION,
        "data" => V1LayerParameter_LayerType::DATA,
        "dropout" => V1LayerParameter_LayerType::DROPOUT,
        "euclidean_loss" => V1LayerParameter_LayerType::EUCLIDEAN_LOSS,
        "flatten" => V1LayerParameter_LayerType::FLATTEN,
        "hdf5_data" => V1LayerParameter_LayerType::HDF5_DATA,
        "hdf5_output" => V1LayerParameter_LayerType::HDF5_OUTPUT,
        "im2col" => V1LayerParameter_LayerType::IM2COL,
        "images" => V1LayerParameter_LayerType::IMAGE_DATA,
        "infogain_loss" => V1LayerParameter_LayerType::INFOGAIN_LOSS,
        "innerproduct" => V1LayerParameter_LayerType::INNER_PRODUCT,
        "lrn" => V1LayerParameter_LayerType::LRN,
        "multinomial_logistic_loss" => V1LayerParameter_LayerType::MULTINOMIAL_LOGISTIC_LOSS,
        "pool" => V1LayerParameter_LayerType::POOLING,
        "relu" => V1LayerParameter_LayerType::RELU,
        "sigmoid" => V1LayerParameter_LayerType::SIGMOID,
        "softmax" => V1LayerParameter_LayerType::SOFTMAX,
        "softmax_loss" => V1LayerParameter_LayerType::SOFTMAX_LOSS,
        "split" => V1LayerParameter_LayerType::SPLIT,
        "tanh" => V1LayerParameter_LayerType::TANH,
        "window_data" => V1LayerParameter_LayerType::WINDOW_DATA,
        _ => {
            assert!(false, "Unknown layer name: {:?}", ty);
            V1LayerParameter_LayerType::NONE
        }
    }
}

/// Return true if any layer contains deprecated data transformation parameters.
pub fn net_needs_data_upgrade(net_param: &NetParameter) -> bool {
    macro_rules! ret_true_if_param_has {
        ($param:ident) => {
            if $param.has_scale() { return true; }
            if $param.has_mean_file() { return true; }
            if $param.has_crop_size() { return true; }
            if $param.has_mirror() { return true; }
        };
    }

    for layer in net_param.get_layers() {
        let ty = layer.get_field_type();
        match ty {
            V1LayerParameter_LayerType::DATA => {
                let layer_param = layer.get_data_param();
                ret_true_if_param_has!(layer_param);
            }
            V1LayerParameter_LayerType::IMAGE_DATA => {
                let layer_param = layer.get_image_data_param();
                ret_true_if_param_has!(layer_param);
            }
            V1LayerParameter_LayerType::WINDOW_DATA => {
                let layer_param = layer.get_window_data_param();
                ret_true_if_param_has!(layer_param);
            }
            _ => {}
        }
    }

    false
}

/// Perform all necessary transformation to upgrade old transformation fields into
/// a `TransformationParameter`.
pub fn upgrade_net_data_transformation(net_param: &mut NetParameter) {
    macro_rules! convert_layer_transform_param {
        ($idx:ident, $t:tt, $param:tt, $mut_param:tt) => {
            if net_param.get_layers()[$idx].get_field_type() == V1LayerParameter_LayerType::$t {
                let mut_layer = &mut net_param.mut_layers()[$idx];
                if mut_layer.$param().has_scale() {
                    let scale = mut_layer.$param().get_scale();
                    mut_layer.mut_transform_param().set_scale(scale);
                    mut_layer.$mut_param().clear_scale();
                }
                if mut_layer.$param().has_mean_file() {
                    let mf = Chars::from(mut_layer.$param().get_mean_file());
                    mut_layer.mut_transform_param().set_mean_file(mf);
                    mut_layer.$mut_param().clear_mean_file();
                }
                if mut_layer.$param().has_crop_size() {
                    let crop = mut_layer.$param().get_crop_size();
                    mut_layer.mut_transform_param().set_crop_size(crop);
                    mut_layer.$mut_param().clear_crop_size();
                }
                if mut_layer.$param().has_mirror() {
                    let mirror = mut_layer.$param().get_mirror();
                    mut_layer.mut_transform_param().set_mirror(mirror);
                    mut_layer.$mut_param().clear_mirror();
                }
            }
        };
    }

    for i in 0..net_param.get_layers().len() {
        convert_layer_transform_param!(i, DATA, get_data_param, mut_data_param);
        convert_layer_transform_param!(i, IMAGE_DATA, get_image_data_param, mut_image_data_param);
        convert_layer_transform_param!(i, WINDOW_DATA, get_window_data_param, mut_window_data_param);
    }
}

/// Return true if the net contains any layers specified as `V1LayerParameter`
pub fn net_needs_v1_to_v2_upgrade(net_param: &NetParameter) -> bool {
    !net_param.get_layers().is_empty()
}

/// Perform all necessary transformations to upgrade a `NetParameter` with deprecated `V1LayerParameter`.
pub fn upgrade_v1_net(v1_net_param: &NetParameter, net_param: &mut NetParameter) -> bool {
    if !v1_net_param.get_layer().is_empty() {
        assert!(false, "{}", "Refusing to upgrade inconsistent NetParameter input; the definition includes \
            both 'layer' and 'layers' fields. The current format defines 'layer' fields with string \
            type like layer { type: 'Layer' ... } and not layers { type: LAYER ... }. Manually \
            switch the definition to 'layer' format to continue.");
    }

    let mut is_fully_compatible = true;
    net_param.clone_from(v1_net_param);
    net_param.clear_layers();
    net_param.clear_layer();
    let mut i = 0;
    for layer in v1_net_param.get_layers() {
        if !upgrade_v1_layer_parameter(layer, net_param.mut_layer().push_default()) {
            error!("Upgrade of input layer {:?} failed.", i);
            is_fully_compatible = false;
        }
        i += 1;
    }

    is_fully_compatible
}

pub fn upgrade_v1_layer_parameter(v1_layer_param: &V1LayerParameter, layer_param: &mut LayerParameter) -> bool {
    layer_param.clear();
    let mut is_fully_compatible = true;
    for b in v1_layer_param.get_bottom() {
        layer_param.mut_bottom().push(b.clone());
    }
    for t in v1_layer_param.get_top() {
        layer_param.mut_top().push(t.clone());
    }
    if v1_layer_param.has_name() {
        layer_param.set_name(Chars::from(v1_layer_param.get_name()));
    }
    for i in v1_layer_param.get_include() {
        layer_param.mut_include().push_default().clone_from(i);
    }
    for e in v1_layer_param.get_exclude() {
        layer_param.mut_exclude().push_default().clone_from(e);
    }
    if v1_layer_param.has_field_type() {
        layer_param.set_field_type(Chars::from(upgrade_v1_layer_type(v1_layer_param.get_field_type())));
    }
    for b in v1_layer_param.get_blobs() {
        layer_param.mut_blobs().push_default().clone_from(b);
    }
    for i in 0..v1_layer_param.get_param().len() {
        while layer_param.get_param().len() <= i {
            layer_param.mut_param().push_default();
        }
        layer_param.mut_param()[i].set_name(v1_layer_param.get_param()[i].clone());
    }
    for i in 0..v1_layer_param.get_blob_share_mode().len() {
        while layer_param.get_param().len() <= i {
            layer_param.mut_param().push_default();
        }
        let mode = match v1_layer_param.get_blob_share_mode()[i] {
            V1LayerParameter_DimCheckMode::STRICT => ParamSpec_DimCheckMode::STRICT,
            V1LayerParameter_DimCheckMode::PERMISSIVE => ParamSpec_DimCheckMode::PERMISSIVE,
        };
        layer_param.mut_param()[i].set_share_mode(mode);
    }
    for i in 0..v1_layer_param.get_blobs_lr().len() {
        while layer_param.get_param().len() <= i {
            layer_param.mut_param().push_default();
        }
        layer_param.mut_param()[i].set_lr_mult(v1_layer_param.get_blobs_lr()[i]);
    }
    for i in 0..v1_layer_param.get_weight_decay().len() {
        while layer_param.get_param().len() <= i {
            layer_param.mut_param().push_default();
        }
        layer_param.mut_param()[i].set_decay_mult(v1_layer_param.get_weight_decay()[i]);
    }
    for loss in v1_layer_param.get_loss_weight() {
        layer_param.mut_loss_weight().push(*loss);
    }
    if v1_layer_param.has_accuracy_param() {
        layer_param.mut_accuracy_param().clone_from(v1_layer_param.get_accuracy_param());
    }
    if v1_layer_param.has_argmax_param() {
        layer_param.mut_argmax_param().clone_from(v1_layer_param.get_argmax_param());
    }
    if v1_layer_param.has_concat_param() {
        layer_param.mut_concat_param().clone_from(v1_layer_param.get_concat_param());
    }
    if v1_layer_param.has_contrastive_loss_param() {
        layer_param.mut_contrastive_loss_param().clone_from(v1_layer_param.get_contrastive_loss_param());
    }
    if v1_layer_param.has_convolution_param() {
        layer_param.mut_convolution_param().clone_from(v1_layer_param.get_convolution_param());
    }
    if v1_layer_param.has_data_param() {
        layer_param.mut_data_param().clone_from(v1_layer_param.get_data_param());
    }
    if v1_layer_param.has_dropout_param() {
        layer_param.mut_dropout_param().clone_from(v1_layer_param.get_dropout_param());
    }
    if v1_layer_param.has_dummy_data_param() {
        layer_param.mut_dummy_data_param().clone_from(v1_layer_param.get_dummy_data_param());
    }
    if v1_layer_param.has_eltwise_param() {
        layer_param.mut_eltwise_param().clone_from(v1_layer_param.get_eltwise_param());
    }
    if v1_layer_param.has_exp_param() {
        layer_param.mut_exp_param().clone_from(v1_layer_param.get_exp_param());
    }
    if v1_layer_param.has_hdf5_data_param() {
        layer_param.mut_hdf5_data_param().clone_from(v1_layer_param.get_hdf5_data_param());
    }
    if v1_layer_param.has_hdf5_output_param() {
        layer_param.mut_hdf5_output_param().clone_from(v1_layer_param.get_hdf5_output_param());
    }
    if v1_layer_param.has_hinge_loss_param() {
        layer_param.mut_hinge_loss_param().clone_from(v1_layer_param.get_hinge_loss_param());
    }
    if v1_layer_param.has_image_data_param() {
        layer_param.mut_image_data_param().clone_from(v1_layer_param.get_image_data_param());
    }
    if v1_layer_param.has_infogain_loss_param() {
        layer_param.mut_infogain_loss_param().clone_from(v1_layer_param.get_infogain_loss_param());
    }
    if v1_layer_param.has_inner_product_param() {
        layer_param.mut_inner_product_param().clone_from(v1_layer_param.get_inner_product_param());
    }
    if v1_layer_param.has_lrn_param() {
        layer_param.mut_lrn_param().clone_from(v1_layer_param.get_lrn_param());
    }
    if v1_layer_param.has_memory_data_param() {
        layer_param.mut_memory_data_param().clone_from(v1_layer_param.get_memory_data_param());
    }
    if v1_layer_param.has_mvn_param() {
        layer_param.mut_mvn_param().clone_from(v1_layer_param.get_mvn_param());
    }
    if v1_layer_param.has_pooling_param() {
        layer_param.mut_pooling_param().clone_from(v1_layer_param.get_pooling_param());
    }
    if v1_layer_param.has_power_param() {
        layer_param.mut_power_param().clone_from(v1_layer_param.get_power_param());
    }
    if v1_layer_param.has_relu_param() {
        layer_param.mut_relu_param().clone_from(v1_layer_param.get_relu_param());
    }
    if v1_layer_param.has_sigmoid_param() {
        layer_param.mut_sigmoid_param().clone_from(v1_layer_param.get_sigmoid_param());
    }
    if v1_layer_param.has_softmax_param() {
        layer_param.mut_softmax_param().clone_from(v1_layer_param.get_softmax_param());
    }
    if v1_layer_param.has_slice_param() {
        layer_param.mut_slice_param().clone_from(v1_layer_param.get_slice_param());
    }
    if v1_layer_param.has_tanh_param() {
        layer_param.mut_tanh_param().clone_from(v1_layer_param.get_tanh_param());
    }
    if v1_layer_param.has_threshold_param() {
        layer_param.mut_threshold_param().clone_from(v1_layer_param.get_threshold_param());
    }
    if v1_layer_param.has_window_data_param() {
        layer_param.mut_window_data_param().clone_from(v1_layer_param.get_window_data_param());
    }
    if v1_layer_param.has_transform_param() {
        layer_param.mut_transform_param().clone_from(v1_layer_param.get_transform_param());
    }
    if v1_layer_param.has_loss_param() {
        layer_param.mut_loss_param().clone_from(v1_layer_param.get_loss_param());
    }
    if v1_layer_param.has_layer() {
        error!("Input NetParameter has V0 layer -- ignoring.");
        is_fully_compatible = false;
    }

    is_fully_compatible
}

pub fn upgrade_v1_layer_type(ty: V1LayerParameter_LayerType) -> &'static str {
    match ty {
        V1LayerParameter_LayerType::NONE => "",
        V1LayerParameter_LayerType::ABSVAL => "AbsVal",
        V1LayerParameter_LayerType::ACCURACY => "Accuracy",
        V1LayerParameter_LayerType::ARGMAX => "ArgMax",
        V1LayerParameter_LayerType::BNLL => "BNLL",
        V1LayerParameter_LayerType::CONCAT => "Concat",
        V1LayerParameter_LayerType::CONTRASTIVE_LOSS => "ContrastiveLoss",
        V1LayerParameter_LayerType::CONVOLUTION => "Convolution",
        V1LayerParameter_LayerType::DECONVOLUTION => "Deconvolution",
        V1LayerParameter_LayerType::DATA => "Data",
        V1LayerParameter_LayerType::DROPOUT => "Dropout",
        V1LayerParameter_LayerType::DUMMY_DATA => "DummyData",
        V1LayerParameter_LayerType::EUCLIDEAN_LOSS => "EuclideanLoss",
        V1LayerParameter_LayerType::ELTWISE => "Eltwise",
        V1LayerParameter_LayerType::EXP => "Exp",
        V1LayerParameter_LayerType::FLATTEN => "Flatten",
        V1LayerParameter_LayerType::HDF5_DATA => "HDF5Data",
        V1LayerParameter_LayerType::HDF5_OUTPUT => "HDF5Output",
        V1LayerParameter_LayerType::HINGE_LOSS => "HingeLoss",
        V1LayerParameter_LayerType::IM2COL => "Im2col",
        V1LayerParameter_LayerType::IMAGE_DATA => "ImageData",
        V1LayerParameter_LayerType::INFOGAIN_LOSS => "InfogainLoss",
        V1LayerParameter_LayerType::INNER_PRODUCT => "InnerProduct",
        V1LayerParameter_LayerType::LRN => "LRN",
        V1LayerParameter_LayerType::MEMORY_DATA => "MemoryData",
        V1LayerParameter_LayerType::MULTINOMIAL_LOGISTIC_LOSS => "MultinomialLogisticLoss",
        V1LayerParameter_LayerType::MVN => "MVN",
        V1LayerParameter_LayerType::POOLING => "Pooling",
        V1LayerParameter_LayerType::POWER => "Power",
        V1LayerParameter_LayerType::RELU => "ReLU",
        V1LayerParameter_LayerType::SIGMOID => "Sigmoid",
        V1LayerParameter_LayerType::SIGMOID_CROSS_ENTROPY_LOSS => "SigmoidCrossEntropyLoss",
        V1LayerParameter_LayerType::SILENCE => "Silence",
        V1LayerParameter_LayerType::SOFTMAX => "Softmax",
        V1LayerParameter_LayerType::SOFTMAX_LOSS => "SoftmaxWithLoss",
        V1LayerParameter_LayerType::SPLIT => "Split",
        V1LayerParameter_LayerType::SLICE => "Slice",
        V1LayerParameter_LayerType::TANH => "TanH",
        V1LayerParameter_LayerType::WINDOW_DATA => "WindowData",
        V1LayerParameter_LayerType::THRESHOLD => "Threshold",
        // _ => {
        //     assert!(false, "Unknown V1LayerParameter layer type: {:?}", ty);
        //     ""
        // }
    }
}

/// Return true if the Net contains input fields.
pub fn net_needs_input_upgrade(net_param: &NetParameter) -> bool {
    !net_param.get_input().is_empty()
}

/// Perform all necessary transformations to upgrade input fields into layers.
pub fn upgrade_net_input(net_param: &mut NetParameter) {
    // Collect inputs and convert to Input layer definitions. If the NetParameter holds an input
    // alone, without shape/dim, then it's a legacy caffe-model and simple stripping the input
    // field is enough.
    let has_shape = !net_param.get_input_shape().is_empty();
    let has_dim = !net_param.get_input_dim().is_empty();
    if has_shape || has_dim {
        let mut layer_param = LayerParameter::new();
        layer_param.set_name(Chars::from("input"));
        layer_param.set_field_type(Chars::from("input"));
        // Convert input fields into a layer.
        for i in 0..net_param.input.len() {
            layer_param.mut_top().push(net_param.input[i].clone());
            let input_param = layer_param.mut_input_param();
            if has_shape {
                input_param.mut_shape().push_default().clone_from(&net_param.input_shape[i]);
            } else {
                // Turn legacy input dimensions into shape.
                let shape = input_param.mut_shape().push_default();
                let first_dim = i * 4;
                for j in first_dim..(first_dim + 4) {
                    shape.mut_dim().push(net_param.input_dim[j] as i64);
                }
            }
        }
        // Swap input layer to beginning of net to satisfy layer dependencies.
        net_param.layer.insert(0, layer_param);
    }

    // Clear inputs.
    net_param.clear_input();
    net_param.clear_input_shape();
    net_param.clear_input_dim();
}

/// Return true if the net contains batch norm layers with manual local LRs
pub fn net_needs_batch_norm_upgrade(net_param: &NetParameter) -> bool {
    for layer in net_param.get_layer() {
        // Check if BatchNorm layers declare three parameters, as required by the previous
        // BatchNorm layer definition.
        if layer.get_field_type() == "BatchNorm" && layer.get_param().len() == 3 {
            return true;
        }
    }

    false
}

/// Perform all necessary transformations to upgrade batch norm layers.
pub fn upgrade_net_batch_norm(net_param: &mut NetParameter) {
    for i in 0..net_param.get_layer().len() {
        // Check if BatchNorm layers declare three parameters, as required by the previous
        // BatchNorm layer definition.
        if net_param.get_layer()[i].get_field_type() == "BatchNorm" &&
            net_param.get_layer()[i].get_param().len() == 3 {
            // set lr_mult and decay_mult to zero. leave all other param intact.
            for param in net_param.mut_layer()[i].mut_param() {
                param.set_lr_mult(0f32);
                param.set_decay_mult(0f32);
            }
        }
    }
}

/// Return true if the solver contains any old solver_type specified as enums.
pub fn solver_needs_type_upgrade(solver_param: &SolverParameter) -> bool {
    solver_param.has_solver_type()
}

pub fn upgrade_solver_type(solver_param: &mut SolverParameter) -> bool {
    assert!(!solver_param.has_solver_type() || !solver_param.has_field_type(),
            "Failed to upgrade solver: old solver_type field (enum) and new type \
            field (string) cannot be both specified in solver proto text.");
    if solver_param.has_solver_type() {
        let ty = match solver_param.get_solver_type() {
            SolverParameter_SolverType::SGD => "SGD",
            SolverParameter_SolverType::NESTEROV => "Nesterov",
            SolverParameter_SolverType::ADAGRAD => "AdaGrad",
            SolverParameter_SolverType::RMSPROP => "RMSProp",
            SolverParameter_SolverType::ADADELTA => "AdaDelta",
            SolverParameter_SolverType::ADAM => "Adam",
        };
        solver_param.set_field_type(Chars::from(ty));
        solver_param.clear_solver_type();
    } else {
        error!("Warning: solver type already up to date.");
        return false;
    }

    true
}

/// Check for deprecations and upgrade the `SolverParameter` as needed.
pub fn upgrade_solver_as_needed(param_file: &str, param: &mut SolverParameter) -> bool {
    let mut success = true;
    // Try to upgrade old style solver_type enum fields into new string type.
    if solver_needs_type_upgrade(param) {
        info!("Attempting to upgrade input file specified using deprecated 'solver_type' field (enum)': {:?}",
              param_file);
        if !upgrade_solver_type(param) {
            success = false;
            error!("Warning: had one or more problems upgrading SolverType (see above).");
        } else {
            info!("Successfully upgraded file specified using deprecated 'solver_type' field (enum) \
                to 'type' field (string).");
            warn!("Note that future Caffe releases will only support 'type' field (string) for a solver's type.");
        }
    }

    success
}

/// Replaces snapshot_prefix of SolverParameter if it is not specified or is set to directory
fn upgrade_snapshot_prefix_property(param_file: &str, param: &mut SolverParameter) {
    if !param.has_snapshot_prefix() {
        let path = Path::new(param_file).with_extension("");
        param.set_snapshot_prefix(Chars::from(path.to_str().unwrap()));
        info!("snapshot_prefix was not specified and is set to {:?}", param.get_snapshot_prefix());
    } else if std::fs::metadata(param.get_snapshot_prefix()).map_or(false, |m| m.is_dir()) {
        let mut path = PathBuf::from(param.get_snapshot_prefix());
        path.push(Path::new(param_file).file_stem().unwrap());
        param.set_snapshot_prefix(Chars::from(path.to_str().unwrap()));
        info!("snapshot_prefix was a directory and is replaced to {:?}", param.get_snapshot_prefix());
    }
}

/// Read parameters from a file into a `SolverParameter` proto message.
pub fn read_solver_params_from_text_file_or_die(param_file: &str, param: &mut SolverParameter) {
    let r = read_proto_from_text_file(param_file, param);
    assert!(r, "Failed to parse SolverParameter file: {:?}", param_file);
    upgrade_solver_as_needed(param_file, param);
    upgrade_snapshot_prefix_property(param_file, param);
}
