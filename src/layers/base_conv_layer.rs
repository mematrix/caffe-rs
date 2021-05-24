use std::cell::RefCell;

use crate::blob::{BlobType, Blob};
use crate::filler::get_filler;
use crate::layer::{CaffeLayer, LayerImpl, BlobVec, SharedBlob};
use crate::proto::caffe::LayerParameter;
use crate::util::im2col::{im2col_cpu, im2col_nd_cpu, col2im_cpu, col2im_nd_cpu};
use crate::util::math_functions::caffe_set;
use cblas::Transpose;

/// Abstract base class that factors out the BLAS code common to `ConvolutionLayer` and `DeconvolutionLayer`.
///
/// **NOTE**: This struct derived the `Default` trait only for the convenience. **Instance should be
/// constructed only by the `new` function**.
#[derive(Default)]
pub struct BaseConvolutionLayerImpl<T: BlobType> {
    pub layer: LayerImpl<T>,
    pub kernel_shape: Blob<i32>,
    pub stride: Blob<i32>,
    pub pad: Blob<i32>,
    pub dilation: Blob<i32>,
    pub conv_input_shape: Blob<i32>,
    pub col_buffer_shape: Vec<i32>,
    pub output_shape: Vec<i32>,
    pub bottom_shape: Vec<i32>,
    pub num_spatial_axes: i32,
    pub bottom_dim: i32,
    pub top_dim: i32,

    pub channel_axis: i32,
    pub num: i32,
    pub channels: i32,
    pub group: i32,
    pub out_spatial_dim: i32,
    pub weight_offset: i32,
    pub num_output: i32,
    pub bias_term: bool,
    pub is_1x1: bool,
    pub force_nd_im2col: bool,

    // private
    num_kernels_im2col: i32,
    num_kernels_col2im: i32,
    conv_out_channels: i32,
    conv_in_channels: i32,
    conv_out_spatial_dim: i32,
    kernel_dim: i32,
    col_offset: i32,
    output_offset: i32,

    col_buffer: Blob<T>,
    bias_multiplier: Blob<T>,
}

impl<T: BlobType> BaseConvolutionLayerImpl<T> {
    pub fn new(param: &LayerParameter) -> Self {
        BaseConvolutionLayerImpl {
            layer: LayerImpl::new(param),
            ..Default::default()
        }
    }
}

pub trait BaseConvolutionLayer<T: BlobType> {
    fn get_conv_impl(&self) -> &BaseConvolutionLayerImpl<T>;

    fn get_conv_impl_mut(&mut self) -> &mut BaseConvolutionLayerImpl<T>;

    /// reverse_dimensions should return true iff we are implementing deconv, so that
    /// conv helpers know which dimensions are which.
    fn reverse_dimensions(&mut self) -> bool;

    /// Compute height_out and width_out from other parameters.
    fn compute_output_shape(&mut self);
}

impl<T: BlobType> dyn BaseConvolutionLayer<T> {
    pub fn layer_setup(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        // Configure the kernel size, padding, stride, and inputs.
        let rev = self.reverse_dimensions();
        let conv = self.get_conv_impl_mut();
        let b0 = bottom[0].as_ref().borrow();

        let conv_param = conv.layer.layer_param.get_convolution_param();
        conv.force_nd_im2col = conv_param.get_force_nd_im2col();
        conv.channel_axis = b0.canonical_axis_index(conv_param.get_axis()) as i32;
        let first_spatial_axis = conv.channel_axis + 1;
        let num_axes = b0.num_axes();
        conv.num_spatial_axes = num_axes - first_spatial_axis;
        check_ge!(conv.num_spatial_axes, 0);

        let mut spatial_dim_blob_shape = Vec::with_capacity(1);
        spatial_dim_blob_shape.resize(1, std::cmp::max(conv.num_spatial_axes, 1));
        conv.kernel_shape.reshape(&spatial_dim_blob_shape);
        let kernel_shape_data = conv.kernel_shape.mutable_cpu_data();
        if conv_param.has_kernel_h() || conv_param.has_kernel_w() {
            assert_eq!(conv.num_spatial_axes, 2, "kernel_h & kernel_w can only be used for 2D convolution.");
            assert!(conv_param.get_kernel_size().is_empty(),
                    "Either kernel_size or kernel_h/w should be specified; not both.");
            kernel_shape_data[0] = conv_param.get_kernel_h() as i32;
            kernel_shape_data[1] = conv_param.get_kernel_w() as i32;
        } else {
            let kernel_dims = conv_param.get_kernel_size();
            let num_kernel_dims = kernel_dims.len() as i32;
            assert!(num_kernel_dims == 1 || num_kernel_dims == conv.num_spatial_axes,
                    "kernel_size must be specified once, or once per spatial dimension (kernel_size \
                    specified {:?} times; {:?} spatial dims).", num_kernel_dims, conv.num_spatial_axes);
            for i in 0..conv.num_spatial_axes as usize {
                kernel_shape_data[i] = kernel_dims[if num_kernel_dims == 1 { 0usize } else { i }] as i32;
            }
        }
        for i in 0..conv.num_spatial_axes as usize {
            check_gt!(kernel_shape_data[i], 0, "Filter dimensions must be nonzero.");
        }

        // Setup stride dimensions (stride)
        conv.stride.reshape(&spatial_dim_blob_shape);
        let stride_data = conv.stride.mutable_cpu_data();
        if conv_param.has_stride_h() || conv_param.has_stride_w() {
            assert_eq!(conv.num_spatial_axes, 2, "stride_h & stride_w can only be used for 2D convolution.");
            assert!(conv_param.get_stride().is_empty(), "Either stride or stride_h/w should be specified; not both.");
            stride_data[0] = conv_param.get_stride_h() as i32;
            stride_data[1] = conv_param.get_stride_w() as i32;
        } else {
            let stride_dims = conv_param.get_stride();
            let num_stride_dims = stride_dims.len() as i32;
            assert!(num_stride_dims == 0 || num_stride_dims == 1 || num_stride_dims == conv.num_spatial_axes,
                    "stride must be specified once, or once per spatial dimension (stride specified \
                    {:?} times; {:?} spatial dims).", num_stride_dims, conv.num_spatial_axes);
            const K_DEFAULT_STRIDE: i32 = 1;
            for i in 0..conv.num_spatial_axes as usize {
                let s_v = if num_stride_dims == 0 {
                    K_DEFAULT_STRIDE
                } else {
                    stride_dims[if num_stride_dims == 1 { 0usize } else { i }] as i32
                };
                stride_data[i] = s_v;
                check_gt!(s_v, 0, "Stride dimensions must be nonzero.");
            }

        }

        // Setup pad dimensions (pad).
        conv.pad.reshape(&spatial_dim_blob_shape);
        let pad_data = conv.pad.mutable_cpu_data();
        if conv_param.has_pad_h() || conv_param.has_pad_w() {
            assert_eq!(conv.num_spatial_axes, 2, "pad_h & pad_w can only be used for 2D convolution.");
            assert!(conv_param.get_pad().is_empty(), "Either pad or pad_h/w should be specified; not both.");
            pad_data[0] = conv_param.get_pad_h() as i32;
            pad_data[1] = conv_param.get_pad_w() as i32;
        } else {
            let pad_dims = conv_param.get_pad();
            let num_pad_dims = pad_dims.len() as i32;
            assert!(num_pad_dims == 0 || num_pad_dims == 1 || num_pad_dims == conv.num_spatial_axes,
                    "pad must be specified once, or once per spatial dimension (pad specified {:?} \
                    times; {:?} spatial dims).", num_pad_dims, conv.num_spatial_axes);
            const K_DEFAULT_PAD: i32 = 0;
            for i in 0..conv.num_spatial_axes as usize {
                pad_data[i] = if num_pad_dims == 0 {
                    K_DEFAULT_PAD
                } else {
                    pad_dims[if num_pad_dims == 1 { 0usize } else { i }] as i32
                };
            }
        }

        // Setup dilation dimensions (dilation).
        conv.dilation.reshape(&spatial_dim_blob_shape);
        let dilation_data = conv.dilation.mutable_cpu_data();
        {
            let dilation_dims = conv_param.get_dilation();
            let num_dilation_dims = dilation_dims.len() as i32;
            assert!(num_dilation_dims == 0 || num_dilation_dims == 1 || num_dilation_dims == conv.num_spatial_axes,
                    "dilation must be specified once, or once per spatial dimension (dilation \
                    specified {:?} times; {:?} spatial dims).", num_dilation_dims, conv.num_spatial_axes);
            const K_DEFAULT_DILATION: i32 = 1;
            for i in 0..conv.num_spatial_axes as usize {
                dilation_data[i] = if num_dilation_dims == 0 {
                    K_DEFAULT_DILATION
                } else if num_dilation_dims == 1 {
                    dilation_dims[0] as i32
                } else {
                    dilation_dims[i] as i32
                };
            }
        }

        // Special case: im2col is the identity for 1x1 convolution with stride 1
        // and no padding, so flag for skipping the buffer and transformation.
        conv.is_1x1 = true;
        for i in 0..conv.num_spatial_axes as usize {
            conv.is_1x1 &= (kernel_shape_data[i] == 1) && (stride_data[i] == 1) && (pad_data[i] == 0);
            if !conv.is_1x1 {
                break;
            }
        }

        // Configure output channels and groups.
        conv.channels = b0.shape_idx(conv.channel_axis);
        conv.num_output = conv_param.get_num_output() as i32;
        check_gt!(conv.num_output, 0);
        conv.group = conv_param.get_group() as i32;
        assert_eq!(conv.channels % conv.group, 0);
        assert_eq!(conv.num_output % conv.group, 0, "Number of output should be multiples of group.");
        if rev {
            conv.conv_out_channels = conv.channels;
            conv.conv_in_channels = conv.num_output;
        } else {
            conv.conv_out_channels = conv.num_output;
            conv.conv_in_channels = conv.channels;
        }

        // Handle the parameters: weights and biases.
        // - blobs[0] holds the filter weights
        // - blobs[1] holds the biases (optional)
        let mut weight_shape = Vec::with_capacity((conv.num_spatial_axes + 2) as usize);
        weight_shape.push(conv.conv_out_channels);
        weight_shape.push(conv.conv_in_channels / conv.group);
        for i in 0..conv.num_spatial_axes as usize {
            weight_shape.push(kernel_shape_data[i]);
        }
        conv.bias_term = conv_param.get_bias_term();
        let mut bias_shape = Vec::new();
        bias_shape.resize(conv.bias_term as i32 as usize, conv.num_output);
        if !conv.layer.blobs.is_empty() {
            assert_eq!(conv.bias_term as i32 + 1, conv.layer.blobs.len() as i32, "Incorrect number of weight blobs.");
            if weight_shape != *conv.layer.blobs[0].as_ref().borrow().shape() {
                let weight_shaped_blob = Blob::<T>::with_shape(&weight_shape);
                assert!(false, "Incorrect weight shape: expected shape {:?}; instead, shape was {:?}",
                        weight_shaped_blob.shape_string(), conv.layer.blobs[0].as_ref().borrow().shape_string());
            }
            if conv.bias_term && bias_shape != *conv.layer.blobs[1].as_ref().borrow().shape() {
                let bias_shaped_blob = Blob::<T>::with_shape(&bias_shape);
                assert!(false, "Incorrect bias shape: expected shape {:?}; instead, shape was {:?}",
                        bias_shaped_blob.shape_string(), conv.layer.blobs[1].as_ref().borrow().shape_string());
            }
            info!("Skipping parameter initialization");
        } else {
            // Initialize and fill the weights:
            // output channels x input channels per-group x kernel height x kernel width
            let mut blob0 = Blob::<T>::with_shape(&weight_shape);
            let weight_filler = get_filler(conv_param.get_weight_filler());
            weight_filler.fill(&mut blob0);
            conv.layer.blobs.push(SharedBlob::new(RefCell::new(blob0)));

            // If necessary, initialize and fill the biases.
            if conv.bias_term {
                let mut blob1 = Blob::<T>::with_shape(&bias_shape);
                let bias_filler = get_filler(conv_param.get_bias_filler());
                bias_filler.fill(&mut blob1);
                conv.layer.blobs.push(SharedBlob::new(RefCell::new(blob1)));
            }
        }

        conv.kernel_dim = conv.layer.blobs[0].as_ref().borrow().count_range_to_end(1);
        conv.weight_offset = conv.conv_out_channels * conv.kernel_dim / conv.group;
        // Propagate gradients to the parameters (as directed by backward pass).
        conv.layer.param_propagate_down.resize(conv.layer.blobs.len(), true);
    }

    pub fn reshape(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        let b0 = bottom[0].as_ref().borrow();
        let first_spatial_axis;
        {
            let conv = self.get_conv_impl_mut();
            first_spatial_axis = conv.channel_axis + 1;
            assert_eq!(b0.num_axes(), first_spatial_axis + conv.num_spatial_axes, "bottom num_axes may not change.");

            conv.num = b0.count_range(0, conv.channel_axis as usize);
            assert_eq!(b0.shape_idx(conv.channel_axis), conv.channels,
                       "Input size incompatible with convolution kernel.");
            // todo: generalize to handle inputs of different shapes.
            for bottom_id in 1..bottom.len() {
                let bi = bottom[bottom_id].as_ref().borrow();
                assert!(b0.shape().eq(bi.shape()), "shape mismatch - bottom[0]: {:?} vs bottom[{:?}]: {:?}",
                        b0.shape_string(), bottom_id, bi.shape_string());
            }
            conv.bottom_shape = b0.shape().clone();
        }

        self.compute_output_shape();
        let rev = self.reverse_dimensions();

        let conv = self.get_conv_impl_mut();
        // Shape the tops.
        let mut top_shape = Vec::with_capacity((conv.channel_axis + conv.num_spatial_axes + 1) as usize);
        top_shape.extend_from_slice(&b0.shape()[0..conv.channel_axis as usize]);
        top_shape.push(conv.num_output);
        for i in 0..conv.num_spatial_axes as usize {
            top_shape.push(conv.output_shape[i]);
        }
        for top_blob in top {
            top_blob.borrow_mut().reshape(&top_shape);
        }

        if rev {
            conv.conv_out_spatial_dim = b0.count_range_to_end(first_spatial_axis as usize);
        } else {
            conv.conv_out_spatial_dim = top[0].as_ref().borrow().count_range_to_end(first_spatial_axis as usize);
        }
        conv.col_offset = conv.kernel_dim * conv.conv_out_spatial_dim;
        conv.output_offset = conv.conv_out_channels * conv.conv_out_spatial_dim / conv.group;

        // Setup input dimensions (conv_input_shape_).
        let bottom_dim_blob_shape = vec![conv.num_spatial_axes + 1];
        conv.conv_input_shape.reshape(&bottom_dim_blob_shape);
        {
            let conv_input_shape_data = conv.conv_input_shape.mutable_cpu_data();
            for i in 0..(conv.num_spatial_axes + 1) as usize {
                let idx = conv.channel_axis + i as i32;
                if rev {
                    conv_input_shape_data[i] = top[0].as_ref().borrow().shape_idx(idx);
                } else {
                    conv_input_shape_data[i] = b0.shape_idx(idx);
                }
            }
        }

        // The im2col result buffer will only hold one image at a time to avoid
        // overly large memory usage. In the special case of 1x1 convolution
        // it goes lazily unused to save memory.
        conv.col_buffer_shape.clear();
        conv.col_buffer_shape.push(conv.kernel_dim * conv.group);
        for i in 0..conv.num_spatial_axes {
            if rev {
                conv.col_buffer_shape.push(conv.bottom_shape[(conv.channel_axis + i) as usize]);
            } else {
                conv.col_buffer_shape.push(conv.output_shape[i as usize]);
            }
        }
        conv.col_buffer.reshape(&conv.col_buffer_shape);
        conv.bottom_dim = b0.count_range_to_end(conv.channel_axis as usize);
        conv.top_dim = top[0].as_ref().borrow().count_range_to_end(conv.channel_axis as usize);
        conv.num_kernels_im2col = conv.conv_in_channels * conv.conv_out_spatial_dim;
        conv.num_kernels_col2im = if rev { conv.top_dim } else { conv.bottom_dim };

        // Set up the all ones "bias multiplier" for adding biases by BLAS
        conv.out_spatial_dim = top[0].as_ref().borrow().count_range_to_end(first_spatial_axis as usize);
        if conv.bias_term {
            let bias_multiplier_shape = vec![conv.out_spatial_dim];
            conv.bias_multiplier.reshape(&bias_multiplier_shape);
            let count = conv.bias_multiplier.count();
            caffe_set(count, T::from_i32(1), conv.bias_multiplier.mutable_cpu_data());
        }
    }

    #[inline]
    pub fn min_bottom_blobs(&self) -> i32 {
        1
    }

    #[inline]
    pub fn min_top_blobs(&self) -> i32 {
        1
    }

    #[inline]
    pub fn equal_num_bottom_top_blobs(&self) -> bool {
        true
    }

    // Helper functions that abstract away the column buffer and gemm arguments.
    // The last argument in forward_cpu_gemm is so that we can skip the im2col if
    // we just called weight_cpu_gemm with the same input.

    pub fn forward_cpu_gemm(&mut self, input: &[T], weights: &[T], output: &mut [T], skip_im2col: bool) {
        if self.get_conv_impl().is_1x1 {
            if !skip_im2col {
                // SAFETY: partial borrow from self. the `col_buffer` field is not used in
                // the `self.conv_im2col_cpu` function.
                let col_buff = {
                    let buff = self.get_conv_impl_mut().col_buffer.mutable_cpu_data();
                    unsafe { std::slice::from_raw_parts_mut(buff.as_mut_ptr(), buff.len()) }
                };
                self.conv_im2col_cpu(input, col_buff);
            }
        }

        let conv = self.get_conv_impl();
        let col_buff = if conv.is_1x1 { input } else { conv.col_buffer.cpu_data() };
        for g in 0..conv.group {
            T::caffe_cpu_gemm(Transpose::None, Transpose::None, conv.conv_out_channels / conv.group,
                              conv.conv_out_spatial_dim, conv.kernel_dim, T::from_i32(1),
                              &weights[(conv.weight_offset * g) as usize..],
                              &col_buff[(conv.col_offset * g) as usize..], T::from_i32(0),
                              &mut output[(conv.output_offset * g) as usize..]);
        }
    }

    pub fn forward_cpu_bias(&self, output: &mut [T], bias: &[T]) {
        let conv = self.get_conv_impl();
        T::caffe_cpu_gemm(Transpose::None, Transpose::None, conv.num_output, conv.out_spatial_dim, 1,
                          T::from_i32(1), bias, conv.bias_multiplier.cpu_data(),
                          T::from_i32(1), output);
    }

    pub fn backward_cpu_gemm(&mut self, input: &[T], weights: &[T], output: &mut [T]) {
        {
            let conv = self.get_conv_impl_mut();
            let col_buff = if conv.is_1x1 {
                // SAFETY: Borrow `output` only once, either this position or the later `conv_col2im_cpu` call.
                unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr(), output.len()) }
            } else {
                conv.col_buffer.mutable_cpu_data()
            };
            for g in 0..conv.group {
                T::caffe_cpu_gemm(Transpose::Ordinary, Transpose::None, conv.kernel_dim,
                                  conv.conv_out_spatial_dim, conv.conv_out_channels / conv.group,
                                  T::from_i32(1), &weights[(conv.weight_offset * g) as usize..],
                                  &input[(conv.output_offset * g) as usize..],
                                  T::from_i32(0), &mut col_buff[(conv.col_offset * g) as usize..]);
            }
        }

        let conv = self.get_conv_impl();
        if !conv.is_1x1 {
            self.conv_col2im_cpu(conv.col_buffer.cpu_data(), output);
        }
    }

    pub fn weight_cpu_gemm(&mut self, input: &[T], output: &[T], weights: &mut [T]) {
        if !self.get_conv_impl().is_1x1 {
            // SAFETY: partial borrow from self. the `col_buffer` field is not used in
            // the `self.conv_im2col_cpu` function.
            let col_buff = {
                let buff = self.get_conv_impl_mut().col_buffer.mutable_cpu_data();
                unsafe { std::slice::from_raw_parts_mut(buff.as_mut_ptr(), buff.len()) }
            };
            self.conv_im2col_cpu(input, col_buff);
        }
        let conv = self.get_conv_impl();
        let col_buff = if conv.is_1x1 { input } else { conv.col_buffer.cpu_data() };
        for g in 0..conv.group {
            T::caffe_cpu_gemm(Transpose::None, Transpose::Ordinary, conv.conv_out_channels / conv.group,
                              conv.kernel_dim, conv.conv_out_spatial_dim, T::from_i32(1),
                              &output[(conv.output_offset * g) as usize..],
                              &col_buff[(conv.col_offset * g) as usize..], T::from_i32(1),
                              &mut weights[(conv.weight_offset * g) as usize..]);
        }
    }

    pub fn backward_cpu_bias(&mut self, bias: &mut [T], input: &[T]) {
        let conv = self.get_conv_impl();
        T::caffe_cpu_gemv(Transpose::None, conv.num_output, conv.out_spatial_dim, T::from_i32(1),
                          input, conv.bias_multiplier.cpu_data(), T::from_i32(1), bias);
    }

    pub fn input_shape(&self, i: i32) -> i32 {
        let conv = self.get_conv_impl();
        conv.bottom_shape[(conv.channel_axis + i) as usize]
    }

    // wrap im2col/col2im so we don't have to remember the (long) argument lists

    fn conv_im2col_cpu(&self, data: &[T], col_buff: &mut [T]) {
        let conv = self.get_conv_impl();
        let conv_input_shape = conv.conv_input_shape.cpu_data();
        let kernel_shape = conv.kernel_shape.cpu_data();
        let pad = conv.pad.cpu_data();
        let stride = conv.stride.cpu_data();
        let dilation = conv.dilation.cpu_data();
        if !conv.force_nd_im2col && conv.num_spatial_axes == 2 {
            im2col_cpu(data, conv.conv_in_channels, conv_input_shape[1], conv_input_shape[2],
                       kernel_shape[0], kernel_shape[1],
                       pad[0], pad[1],
                       stride[0], stride[1],
                       dilation[0], dilation[1], col_buff);
        } else {
            let col_buffer_shape = conv.col_buffer_shape.as_slice();
            im2col_nd_cpu(data, conv.num_spatial_axes as usize, conv_input_shape,
                          col_buffer_shape, kernel_shape, pad, stride, dilation, col_buff);
        }
    }

    fn conv_col2im_cpu(&self, col_buff: &[T], data: &mut [T]) {
        let conv = self.get_conv_impl();
        let conv_input_shape = conv.conv_input_shape.cpu_data();
        let kernel_shape = conv.kernel_shape.cpu_data();
        let pad = conv.pad.cpu_data();
        let stride = conv.stride.cpu_data();
        let dilation = conv.dilation.cpu_data();
        if !conv.force_nd_im2col && conv.num_spatial_axes == 2 {
            col2im_cpu(col_buff, conv.conv_in_channels, conv_input_shape[1], conv_input_shape[2],
                       kernel_shape[0], kernel_shape[1],
                       pad[0], pad[1],
                       stride[0], stride[1],
                       dilation[0], dilation[1], data);
        } else {
            let col_buffer_shape = conv.col_buffer_shape.as_slice();
            col2im_nd_cpu(col_buff, conv.num_spatial_axes as usize, conv_input_shape,
                          col_buffer_shape, kernel_shape, pad, stride, dilation, data);
        }
    }
}
