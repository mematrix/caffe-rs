use crate::blob::BlobType;
use crate::util::math_functions::caffe_set;


/// Function uses casting from int to unsigned to compare if value of
/// parameter a is greater or equal to zero and lower than value of
/// parameter b. The b parameter is of type signed and is always positive,
/// therefore its value is always lower than `0x800...` where casting
/// negative value of a parameter converts it to value higher than `0x800...`
/// The casting allows to use one condition instead of two.
fn is_a_ge_zero_and_a_lt_b(a: i32, b: i32) -> bool {
    (a as u32) < (b as u32)
}

fn im2col_nd_core_cpu<T: BlobType>(data_input: &[T], im2col: bool, num_spatial_axes: usize, im_shape: &[i32],
                                   col_shape: &[i32], kernel_shape: &[i32], pad: &[i32], stride: &[i32],
                                   dilation: &[i32], data_output: &mut [T]) {
    if !im2col {
        let mut im_size = im_shape[0];
        for i in 0..num_spatial_axes {
            im_size *= im_shape[i + 1];
        }
        caffe_set(im_size as usize, T::default(), data_output);
    }

    let mut kernel_size = 1;
    for i in 0..num_spatial_axes {
        kernel_size *= kernel_shape[i];
    }

    let channels_col = col_shape[0];
    let mut d_offset = Vec::with_capacity(num_spatial_axes);
    d_offset.resize(num_spatial_axes, 0);
    let mut d_iter = Vec::with_capacity(num_spatial_axes);
    d_iter.resize(num_spatial_axes, 0);
    for c_col in 0..channels_col {
        // Loop over spatial axes in reverse order to compute a  per-axis offset.
        let mut offset = c_col;
        for d_i in (0..num_spatial_axes).rev() {
            if d_i < num_spatial_axes - 1 {
                offset /= kernel_shape[d_i + 1];
            }
            d_offset[d_i] = offset % kernel_shape[d_i];
        }

        let mut incremented = true;
        while incremented {
            // Loop over spatial axes in forward order to compute the indices in the
            // image and column, and whether the index lies in the padding.
            let mut index_col = c_col;
            let mut index_im = c_col / kernel_size;
            let mut is_padding = false;
            for d_i in 0..num_spatial_axes {
                let d = d_iter[d_i];
                let d_im = d * stride[d_i] - pad[d_i] + d_offset[d_i] * dilation[d_i];
                is_padding |= (d_im < 0) || (d_im >= im_shape[d_i + 1]);
                index_col *= col_shape[d_i + 1];
                index_col += d;
                index_im *= im_shape[d_i + 1];
                index_im += d_im;
            }

            if im2col {
                if is_padding {
                    data_output[index_col as usize] = T::default();
                } else {
                    data_output[index_col as usize] = data_input[index_im as usize];
                }
            } else if !is_padding {
                // col2im
                data_output[index_im as usize] = data_input[index_col as usize];
            }

            // Loop over spatial axes in reverse order to choose an index, like counting.
            incremented = false;
            for d_i in (0..num_spatial_axes).rev() {
                let d_max = col_shape[d_i + 1];
                check_lt!(d_iter[d_i], d_max);
                if d_iter[d_i] == d_max - 1 {
                    d_iter[d_i] = 0;
                } else {
                    // d_iter[d_i] < d_max - 1
                    d_iter[d_i] += 1;
                    incremented = true;
                    break;
                }
            }
        }
    }
}

pub fn im2col_nd_cpu<T: BlobType>(data_im: &[T], num_spatial_axes: usize, im_shape: &[i32],
                                  col_shape: &[i32], kernel_shape: &[i32], pad: &[i32], stride: &[i32],
                                  dilation: &[i32], data_col: &mut [T]) {
    const K_IM2COL: bool = true;
    im2col_nd_core_cpu(data_im, K_IM2COL, num_spatial_axes, im_shape, col_shape,
                       kernel_shape, pad, stride, dilation, data_col);
}

pub fn im2col_cpu<T: BlobType>(data_im: &[T], channels: i32, height: i32, width: i32, kernel_h: i32, kernel_w: i32,
                               pad_h: i32, pad_w: i32, stride_h: i32, stride_w: i32,
                               dilation_h: i32, dilation_w: i32, data_col: &mut [T]) {
    let output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    let output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    let channel_size = (height * width) as usize;

    let mut channel_offset = 0usize;
    let mut col_offset = 0usize;
    for _channel in 0..channels {
        let data_im = &data_im[channel_offset..];
        for kernel_row in 0..kernel_h {
            for kernel_col in 0..kernel_w {
                let mut input_row = -pad_h + kernel_row * dilation_h;
                for _output_rows in 0..output_h {
                    if !is_a_ge_zero_and_a_lt_b(input_row, height) {
                        for _output_cols in 0..output_w {
                            data_col[col_offset] = T::default();
                            col_offset += 1;
                        }
                    } else {
                        let mut input_col = -pad_w + kernel_col * dilation_w;
                        for _output_col in 0..output_w {
                            if is_a_ge_zero_and_a_lt_b(input_col, width) {
                                data_col[col_offset] = data_im[(input_row * width + input_col) as usize];
                                col_offset += 1;
                            } else {
                                data_col[col_offset] = T::default();
                                col_offset += 1;
                            }
                            input_col += stride_w;
                        }
                    }

                    input_row += stride_h;
                }
            }
        }

        channel_offset += channel_size;
    }
}

pub fn col2im_nd_cpu<T: BlobType>(data_col: &[T], num_spatial_axes: usize, im_shape: &[i32],
                                  col_shape: &[i32], kernel_shape: &[i32], pad: &[i32], stride: &[i32],
                                  dilation: &[i32], data_im: &mut [T]) {
    const K_IM2COL: bool = false;
    im2col_nd_core_cpu(data_col, K_IM2COL, num_spatial_axes, im_shape, col_shape,
                       kernel_shape, pad, stride, dilation, data_im);
}

pub fn col2im_cpu<T: BlobType>(data_col: &[T], channels: i32, height: i32, width: i32, kernel_h: i32, kernel_w: i32,
                               pad_h: i32, pad_w: i32, stride_h: i32, stride_w: i32,
                               dilation_h: i32, dilation_w: i32, data_im: &mut [T]) {
    caffe_set((height * width * channels) as usize, T::default(), data_im);
    let output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    let output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    let output_w = output_w as usize;
    let channel_size = (height * width) as usize;
    let mut channel_offset = 0usize;
    let mut col_offset = 0usize;
    for _channel in 0..channels {
        let data_im = &mut data_im[channel_offset..];
        for kernel_row in 0..kernel_h {
            for kernel_col in 0..kernel_w {
                let mut input_row = -pad_h + kernel_row * dilation_h;
                for _output_rows in 0..output_h {
                    if !is_a_ge_zero_and_a_lt_b(input_row, height) {
                        col_offset += output_w;
                    } else {
                        let mut input_col = -pad_w + kernel_col * dilation_w;
                        for _output_col in 0..output_w {
                            if is_a_ge_zero_and_a_lt_b(input_col, width) {
                                data_im[(input_row * width + input_col) as usize] += data_col[col_offset];
                            }
                            col_offset += 1;
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }

        channel_offset += channel_size;
    }
}
