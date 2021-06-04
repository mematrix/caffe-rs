use std::rc::Rc;

use cblas::Transpose;

use crate::blob::{BlobType, Blob};
use crate::layer::{LayerImpl, CaffeLayer, BlobVec, def_layer_setup, SharedBlob, make_shared_blob};
use crate::proto::caffe::{LayerParameter, Phase};
use crate::util::math_functions::{caffe_set, caffe_copy};


pub struct BatchNormLayer<T: BlobType> {
    layer: LayerImpl<T>,
    mean: Blob<T>,
    variance: Blob<T>,
    temp: Blob<T>,
    x_norm: Blob<T>,
    use_global_stats: bool,
    moving_average_fraction: T,
    channels: i32,
    eps: T,

    // extra temporary variables is used to carry out sums/broadcasting using BLAS
    batch_sum_multiplier: Blob<T>,
    num_by_chans: Blob<T>,
    spatial_sum_multiplier: Blob<T>,
}

impl<T: BlobType> BatchNormLayer<T> {
    pub fn new(param: &LayerParameter) -> Self {
        Self {
            layer: LayerImpl::new(param),
            mean: Blob::new(),
            variance: Blob::new(),
            temp: Blob::new(),
            x_norm: Blob::new(),
            use_global_stats: false,
            moving_average_fraction: T::default(),
            channels: 0,
            eps: T::default(),
            batch_sum_multiplier: Blob::new(),
            num_by_chans: Blob::new(),
            spatial_sum_multiplier: Blob::new(),
        }
    }

    fn backward_cpu_impl(&mut self, top_diff: &[T], bottom_diff: &mut [T], num: i32, spatial_dim: i32) {
        if self.use_global_stats {
            T::caffe_div(self.temp.count(), top_diff, self.temp.cpu_data(), bottom_diff);
            return;
        }

        let top_data = self.x_norm.cpu_data();
        // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
        //
        // dE(Y)/dX =
        //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
        //     ./ sqrt(var(X) + eps)
        //
        // where \cdot and ./ are hadamard product and elementwise division,
        // respectively, dE/dY is the top diff, and mean/var/sum are all computed
        // along all dimensions except the channels dimension.  In the above
        // equation, the operations allow for expansion (i.e. broadcast) along all
        // dimensions except the channels dimension where required.

        // sum(dE/dY \cdot Y)
        T::caffe_mul(self.temp.count(), top_data, top_diff, bottom_diff);
        T::caffe_cpu_gemv(Transpose::None, self.channels * num,  spatial_dim, T::from_i32(1),
                          bottom_diff, self.spatial_sum_multiplier.cpu_data(), T::default(),
                          self.num_by_chans.mutable_cpu_data());
        T::caffe_cpu_gemv(Transpose::Ordinary, num, self.channels, T::from_i32(1),
                          self.num_by_chans.cpu_data(), self.batch_sum_multiplier.cpu_data(),
                          T::default(), self.mean.mutable_cpu_data());

        // reshape (broadcast) the above
        T::caffe_cpu_gemm(Transpose::None, Transpose::None, num, self.channels, 1, T::from_i32(1),
                          self.batch_sum_multiplier.cpu_data(), self.mean.cpu_data(), T::default(),
                          self.num_by_chans.mutable_cpu_data());
        T::caffe_cpu_gemm(Transpose::None, Transpose::None, self.channels * num, spatial_dim, 1,
                          T::from_i32(1), self.num_by_chans.cpu_data(),
                          self.spatial_sum_multiplier.cpu_data(), T::default(), bottom_diff);

        // sum(dE/dY \cdot Y) \cdot Y
        T::caffe_mul_assign(self.temp.count(), bottom_diff, top_data);

        // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
        T::caffe_cpu_gemv(Transpose::None, self.channels * num, spatial_dim, T::from_i32(1),
                          top_diff, self.spatial_sum_multiplier.cpu_data(), T::default(),
                          self.num_by_chans.mutable_cpu_data());
        T::caffe_cpu_gemv(Transpose::Ordinary, num, self.channels, T::from_i32(1),
                          self.num_by_chans.cpu_data(), self.batch_sum_multiplier.cpu_data(), T::default(),
                          self.mean.mutable_cpu_data());

        // reshape (broadcast) the above to make
        // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
        T::caffe_cpu_gemm(Transpose::None, Transpose::None, num, self.channels, 1, T::from_i32(1),
                          self.batch_sum_multiplier.cpu_data(), self.mean.cpu_data(), T::default(),
                          self.num_by_chans.mutable_cpu_data());
        T::caffe_cpu_gemm(Transpose::None, Transpose::None, self.channels * num, spatial_dim, 1,
                          T::from_i32(1), self.num_by_chans.cpu_data(),
                          self.spatial_sum_multiplier.cpu_data(), T::from_i32(1), bottom_diff);

        // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
        T::caffe_cpu_axpby(self.temp.count() as i32, T::from_i32(1), top_diff,
                           T::from_f64(-1f64 / (num * spatial_dim) as f64), bottom_diff);

        // note: self.temp still contains sqrt(var(X)+eps), computed during the forward pass.
        // SAFETY: `caffe_div` do an element-wise op of the slice items separately.
        let im_bottom_diff = unsafe {
            std::slice::from_raw_parts(bottom_diff.as_ptr(), bottom_diff.len())
        };
        T::caffe_div(self.temp.count(), im_bottom_diff, self.temp.cpu_data(), bottom_diff);
    }
}

impl<T: BlobType> CaffeLayer for BatchNormLayer<T> {
    type DataType = T;

    fn get_impl(&self) -> &LayerImpl<Self::DataType> {
        &self.layer
    }

    fn get_impl_mut(&mut self) -> &mut LayerImpl<Self::DataType> {
        &mut self.layer
    }

    fn layer_setup(&mut self, bottom: &BlobVec<Self::DataType>, _top: &BlobVec<Self::DataType>) {
        let param = self.layer.layer_param.get_batch_norm_param();
        self.moving_average_fraction = T::from_f32(param.get_moving_average_fraction());
        self.use_global_stats = self.layer.phase == Phase::TEST;
        if param.has_use_global_stats() {
            self.use_global_stats = param.get_use_global_stats();
        }

        let b0 = &bottom[0];
        if b0.as_ref().borrow().num_axes() == 1 {
            self.channels = 1;
        } else {
            self.channels = b0.as_ref().borrow().shape_idx(1);
        }
        self.eps = T::from_f32(param.get_eps());

        if !self.layer.blobs.is_empty() {
            info!("Skipping parameter initialization.");
        } else {
            self.layer.blobs.reserve(3);
            let mut sz = vec![self.channels];
            self.layer.blobs.push(make_shared_blob(Blob::with_shape(&sz)));
            self.layer.blobs.push(make_shared_blob(Blob::with_shape(&sz)));
            sz[0] = 1;
            self.layer.blobs.push(make_shared_blob(Blob::with_shape(&sz)));
            for blob in &self.layer.blobs {
                let mut blob = blob.borrow_mut();
                let count = blob.count();
                caffe_set(count, T::default(), blob.mutable_cpu_data());
            }
        }

        // Mask statistics from optimization by setting local learning rates
        // for mean, variance, and the bias correction to zero.
        for i in 0..self.layer.blobs.len() {
            if self.layer.layer_param.get_param().len() == i {
                let fixed_param_spec = self.layer.layer_param.mut_param().push_default();
                fixed_param_spec.set_lr_mult(0f32);
            } else {
                assert_eq!(self.layer.layer_param.get_param()[i].get_lr_mult(), 0f32,
                           "Cannot configure batch normalization statistics as layer parameters.");
            }
        }
    }

    fn reshape(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        let b0 = bottom[0].as_ref().borrow();
        if b0.num_axes() >= 1 {
            assert_eq!(b0.shape_idx(1), self.channels);
        }
        top[0].borrow_mut().reshape_like(&*b0);

        let mut sz = vec![self.channels];
        self.mean.reshape(&sz);
        self.variance.reshape(&sz);
        self.temp.reshape(&sz);
        self.x_norm.reshape(&sz);
        sz[0] = b0.shape_idx(0);
        self.batch_sum_multiplier.reshape(&sz);

        let spatial_dim = b0.count() as i32 / (self.channels * b0.shape_idx(0));
        if self.spatial_sum_multiplier.num_axes() == 0 || self.spatial_sum_multiplier.shape_idx(0) != spatial_dim {
            sz[0] = spatial_dim;
            self.spatial_sum_multiplier.reshape(&sz);
            let count = self.spatial_sum_multiplier.count();
            let multiplier_data = self.spatial_sum_multiplier.mutable_cpu_data();
            caffe_set(count, T::from_i32(1), multiplier_data);
        }

        let num_by_chans = self.channels * b0.shape_idx(0);
        if self.num_by_chans.num_axes() == 0 || self.num_by_chans.shape_idx(0) != num_by_chans {
            sz[0] = num_by_chans;
            self.num_by_chans.reshape(&sz);
            let count = self.batch_sum_multiplier.count();
            caffe_set(count, T::from_i32(1), self.batch_sum_multiplier.mutable_cpu_data());
        }
    }

    fn layer_type(&self) -> &'static str {
        "BatchNorm"
    }

    fn exact_num_bottom_blobs(&self) -> i32 {
        1
    }

    fn exact_num_top_blobs(&self) -> i32 {
        1
    }

    fn forward_cpu(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        let mut t0 = top[0].borrow_mut();
        let t0_count = t0.count();

        let (num, count, spatial_dim, top_data) = if !Rc::ptr_eq(&bottom[0], &top[0]) {
            let b0 = bottom[0].as_ref().borrow();
            let num = b0.shape_idx(0);
            let count = b0.count();
            let spatial_dim = count as i32 / (num * self.channels);
            let top_data = t0.mutable_cpu_data();
            caffe_copy(count, b0.cpu_data(), top_data);
            (num, count, spatial_dim, top_data)
        } else {
            let num = t0.shape_idx(0);
            let count = t0.count();
            let spatial_dim = count as i32 / (num * self.channels);
            (num, count, spatial_dim, t0.mutable_cpu_data())
        };

        if self.use_global_stats {
            // use the stored mean/variance estimates.
            let scale_factor = self.layer.blobs[2].as_ref().borrow().cpu_data()[0];
            let scale_factor = if scale_factor.is_zero() {
                scale_factor
            } else {
                let mut r = T::from_i32(1);
                r /= scale_factor;
                r
            };
            let n = self.variance.count() as i32;
            T::caffe_cpu_scale(n, scale_factor, self.layer.blobs[0].as_ref().borrow().cpu_data(),
                               self.mean.mutable_cpu_data());
            T::caffe_cpu_scale(n, scale_factor, self.layer.blobs[1].as_ref().borrow().cpu_data(),
                               self.variance.mutable_cpu_data());
        } else {
            // compute mean
            if Rc::ptr_eq(&bottom[0], &top[0]) {
                T::caffe_cpu_gemv(Transpose::None, self.channels * num, spatial_dim,
                                  T::from_f64(1f64 / (num * spatial_dim) as f64), top_data,
                                  self.spatial_sum_multiplier.cpu_data(), T::default(),
                                  self.num_by_chans.mutable_cpu_data());
            } else {
                let b0 = bottom[0].as_ref().borrow();
                T::caffe_cpu_gemv(Transpose::None, self.channels * num, spatial_dim,
                                  T::from_f64(1f64 / (num * spatial_dim) as f64), b0.cpu_data(),
                                  self.spatial_sum_multiplier.cpu_data(), T::default(),
                                  self.num_by_chans.mutable_cpu_data());
            }
            T::caffe_cpu_gemv(Transpose::Ordinary, num, self.channels, T::from_i32(1),
                              self.num_by_chans.cpu_data(), self.batch_sum_multiplier.cpu_data(), T::default(),
                              self.mean.mutable_cpu_data());
        }

        // subtract mean
        T::caffe_cpu_gemm(Transpose::None, Transpose::None, num, self.channels, 1, T::from_i32(1),
                          self.batch_sum_multiplier.cpu_data(), self.mean.cpu_data(), T::default(),
                          self.num_by_chans.mutable_cpu_data());
        T::caffe_cpu_gemm(Transpose::None, Transpose::None, self.channels * num, spatial_dim, 1,
                          T::from_i32(-1), self.num_by_chans.cpu_data(), self.spatial_sum_multiplier.cpu_data(),
                          T::from_i32(1), top_data);

        if !self.use_global_stats {
            // compute variance using var(X) = E((X-EX)^2)
            T::caffe_sqr(t0_count, top_data, self.temp.mutable_cpu_data()); // (X-EX)^2
            T::caffe_cpu_gemv(Transpose::None, self.channels * num, spatial_dim,
                              T::from_f64(1f64 / (num * spatial_dim) as f64), self.temp.cpu_data(),
                              self.spatial_sum_multiplier.cpu_data(), T::default(),
                              self.num_by_chans.mutable_cpu_data());
            T::caffe_cpu_gemv(Transpose::Ordinary, num, self.channels, T::from_i32(1),
                              self.num_by_chans.cpu_data(), self.batch_sum_multiplier.cpu_data(), T::default(),
                              self.variance.mutable_cpu_data());    // E((X_EX)^2)

            // compute and save moving average
            let mut blob = self.layer.blobs[2].borrow_mut();
            let blob = blob.mutable_cpu_data();
            blob[0] *= self.moving_average_fraction;
            blob[0] += T::from_i32(1);
            T::caffe_cpu_axpby(self.mean.count() as i32, T::from_i32(1), self.mean.cpu_data(),
                               self.moving_average_fraction,
                               self.layer.blobs[0].as_ref().borrow_mut().mutable_cpu_data());
            let m = count as i32 / self.channels;
            let bias_correction_factor = if m > 1 {
                let mut t = T::from_i32(m);
                t /= T::from_i32(m - 1);
                t
            } else {
                T::from_i32(1)
            };
            T::caffe_cpu_axpby(self.variance.count() as i32, bias_correction_factor, self.variance.cpu_data(),
                               self.moving_average_fraction,
                               self.layer.blobs[0].as_ref().borrow_mut().mutable_cpu_data());
        }

        // normalize variance
        T::caffe_add_scalar(self.variance.count(), self.eps, self.variance.mutable_cpu_data());
        {
            // SAFETY: the `caffe_sqrt` do an element-wise op on each item of slice separately.
            let mut_data = unsafe {
                let d = self.variance.mutable_cpu_data();
                std::slice::from_raw_parts_mut(d.as_mut_ptr(), d.len())
            };
            T::caffe_sqrt(self.variance.count(), self.variance.cpu_data(), mut_data);
        }

        // replicate variance to input size
        T::caffe_cpu_gemm(Transpose::None, Transpose::None, num, self.channels, 1, T::from_i32(1),
                          self.batch_sum_multiplier.cpu_data(), self.variance.cpu_data(), T::default(),
                          self.num_by_chans.mutable_cpu_data());
        T::caffe_cpu_gemm(Transpose::None, Transpose::None, self.channels * num, spatial_dim, 1,
                          T::from_i32(1), self.num_by_chans.cpu_data(), self.spatial_sum_multiplier.cpu_data(),
                          T::default(), self.temp.mutable_cpu_data());
        // SAFETY: the `caffe_div` do an element-wise op on each item of slice separately.
        let im_top_data = unsafe {
            std::slice::from_raw_parts(top_data.as_ptr(), top_data.len())
        };
        T::caffe_div(self.temp.count(), im_top_data, self.temp.cpu_data(), top_data);

        caffe_copy(self.x_norm.count(), top_data, self.x_norm.mutable_cpu_data());
    }

    fn forward_gpu(&mut self, bottom: &BlobVec<Self::DataType>, top: &BlobVec<Self::DataType>) {
        no_gpu!();
    }

    fn backward_cpu(&mut self, top: &BlobVec<Self::DataType>, _propagate_down: &Vec<bool>, bottom: &BlobVec<Self::DataType>) {
        if Rc::ptr_eq(&bottom[0], &top[0]) {
            let mut t0 = top[0].borrow_mut();
            let count = self.x_norm.count();
            caffe_copy(count, t0.cpu_diff(), self.x_norm.mutable_cpu_diff());
            // SAFETY: `backward_cpu_impl()` does not access the `self.x_norm.cpu_diff()` memory.
            let top_diff = unsafe {
                let diff = self.x_norm.cpu_diff();
                std::slice::from_raw_parts(diff.as_ptr(), diff.len())
            };
            let num = t0.shape()[0];
            let spatial_dim = t0.count() as i32 / (t0.shape_idx(0) * self.channels);
            let bottom_diff = t0.mutable_cpu_diff();
            self.backward_cpu_impl(top_diff, bottom_diff, num, spatial_dim);
        } else {
            let t0 = top[0].as_ref().borrow();
            let mut b0 = bottom[0].borrow_mut();
            let top_diff = t0.cpu_diff();
            let num = b0.shape()[0];
            let spatial_dim = b0.count() as i32 / (b0.shape_idx(0) * self.channels);
            let bottom_diff = b0.mutable_cpu_diff();
            self.backward_cpu_impl(top_diff, bottom_diff, num, spatial_dim);
        }
    }

    fn backward_gpu(&mut self, top: &BlobVec<Self::DataType>, propagate_down: &Vec<bool>,
                    bottom: &BlobVec<Self::DataType>) {
        no_gpu!();
    }
}

register_layer_class!(BatchNorm);
