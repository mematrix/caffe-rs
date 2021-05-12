use std::collections::HashMap;

use protobuf::{Chars, Clear};

use crate::proto::caffe::{NetParameter, LayerParameter};


/// Copy NetParameters with SplitLayers added to replace any shared bottom
/// blobs with unique bottom blobs provided by the SplitLayer.
pub fn insert_splits(param: &NetParameter, param_split: &mut NetParameter) {
    // Initialize by copying from the input NetParameter.
    param_split.clone_from(param);
    param_split.clear_layer();

    let mut blob_name_to_last_top_idx = HashMap::<String, (usize, usize)>::new();
    let mut bottom_idx_to_source_top_idx = HashMap::<(usize, usize), (usize, usize)>::new();
    let mut top_idx_to_bottom_count = HashMap::<(usize, usize), usize>::new();
    let mut top_idx_to_loss_weight = HashMap::<(usize, usize), f32>::new();
    let mut top_idx_to_bottom_split_idx = HashMap::<(usize, usize), usize>::new();
    let mut layer_idx_to_layer_name = HashMap::<usize, String>::new();

    for i in 0..param.get_layer().len() {
        let layer_param = &param.get_layer()[i];
        layer_idx_to_layer_name.insert(i, layer_param.get_name().to_string());
        for j in 0..layer_param.get_bottom().len() {
            let blob_name: &str = layer_param.get_bottom()[j].as_ref();
            if !blob_name_to_last_top_idx.contains_key(blob_name) {
                assert!(false, "Unknown bottom blob '{:?}' (layer '{:?}', bottom index {:?})",
                        blob_name, layer_param.get_name(), j);
            }

            let top_idx = blob_name_to_last_top_idx[blob_name];
            bottom_idx_to_source_top_idx.insert((i, j), top_idx);
            *top_idx_to_bottom_count.entry(top_idx).or_default() += 1;
        }
        for j in 0..layer_param.get_top().len() {
            let blob_name: &str = layer_param.get_top()[j].as_ref();
            blob_name_to_last_top_idx.insert(blob_name.to_string(), (i, j));
        }
        // A use of a top blob as a loss should be handled similarly to the use of a top blob as
        // a bottom blob to another layer.
        let last_loss = std::cmp::min(layer_param.get_loss_weight().len(), layer_param.get_top().len());
        for j in 0..last_loss {
            let blob_name: &str = layer_param.get_top()[j].as_ref();
            let top_idx = blob_name_to_last_top_idx[blob_name];
            let loss = layer_param.get_loss_weight()[j];
            top_idx_to_loss_weight.insert(top_idx, loss);
            if loss != 0f32 {
                *top_idx_to_bottom_count.entry(top_idx).or_default() += 1;
            }
        }
    }

    for i in 0..param.get_layer().len() {
        let layer_top_len;
        let layer_idx;

        {
            layer_idx = param_split.get_layer().len();
            let layer_param = param_split.mut_layer().push_default();
            layer_param.clone_from(&param.get_layer()[i]);
            layer_top_len = layer_param.get_top().len();

            // Replace any shared bottom blobs with split layer outputs.
            for j in 0..layer_param.get_bottom().len() {
                let top_idx = bottom_idx_to_source_top_idx[&(i, j)];
                let split_count = top_idx_to_bottom_count[&top_idx];
                if split_count > 1 {
                    let layer_name = layer_idx_to_layer_name[&top_idx.0].as_str();
                    let split_idx = top_idx_to_bottom_split_idx.entry(top_idx).or_default();
                    let value = split_blob_name(layer_name, layer_param.get_bottom()[j].as_ref(),
                                                top_idx.1, *split_idx);
                    *split_idx += 1;
                    layer_param.mut_bottom()[j] = Chars::from(value);
                }
            }
        }

        // Create split layer for any top blobs used by other layers as bottom blobs more than once.
        for j in 0..layer_top_len {
            let top_idx = (i, j);
            let split_count = top_idx_to_bottom_count.get(&top_idx).map_or(0usize, |v| *v);
            if split_count > 1 {
                let layer_name = layer_idx_to_layer_name[&i].as_str();
                let mut split_layer_param = LayerParameter::new();
                let loss_weight = top_idx_to_loss_weight.get(&top_idx).map_or(0f32, |v| *v);
                {
                    let blob_name: &str = param_split.get_layer()[layer_idx].get_top()[j].as_ref();
                    configure_split_layer(layer_name, blob_name, j, split_count, loss_weight, &mut split_layer_param);
                }
                param_split.mut_layer().push(split_layer_param);

                if loss_weight != 0f32 {
                    param_split.mut_layer()[layer_idx].clear_loss_weight();
                    *top_idx_to_bottom_split_idx.entry(top_idx).or_default() += 1;
                }
            }
        }
    }
}

pub fn configure_split_layer(layer_name: &str, blob_name: &str, blob_idx: usize, split_count: usize,
                             loss_weight: f32, split_layer_param: &mut LayerParameter) {
    split_layer_param.clear();
    split_layer_param.mut_bottom().push(Chars::from(blob_name));
    split_layer_param.set_name(Chars::from(split_layer_name(layer_name, blob_name, blob_idx)));
    split_layer_param.set_field_type(Chars::from("Split"));

    for k in 0..split_count {
        split_layer_param.mut_top().push(Chars::from(split_blob_name(layer_name, blob_name, blob_idx, k)));
        if loss_weight != 0f32 {
            if k == 0 {
                split_layer_param.mut_loss_weight().push(loss_weight);
            } else {
                split_layer_param.mut_loss_weight().push(0f32);
            }
        }
    }
}

pub fn split_layer_name(layer_name: &str, blob_name: &str, blob_idx: usize) -> String {
    format!("{}_{}_{}_split", blob_name, layer_name, blob_idx)
}

pub fn split_blob_name(layer_name: &str, blob_name: &str, blob_idx: usize, split_idx: usize) -> String {
    format!("{}_{}_{}_split_{}", blob_name, layer_name, blob_idx, split_idx)
}
