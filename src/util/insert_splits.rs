
use crate::proto::caffe::{NetParameter, LayerParameter};


pub fn insert_splits(param: &NetParameter, param_split: &mut NetParameter) {
    todo!("insert_splits");
}

pub fn configure_split_layer(layer_name: &str, blob_name: &str, blob_idx: usize, split_count: usize,
                             loss_weight: f32, split_layer_param: &mut LayerParameter) {
    todo!();
}

pub fn split_layer_name(layer_name: &str, blob_name: &str, blob_idx: usize) -> String {
    todo!()
}

pub fn split_blob_name(layer_name: &str, blob_name: &str, blob_idx: usize, split_idx: usize) -> String {
    todo!()
}
