//! Fillers are random number generators that fills a blob using the specified
//! algorithm. The expectation is that they are only going to be used during
//! initialization time and will not involve any GPUs.

use crate::blob::{BlobType, Blob};
use crate::proto::caffe::{FillerParameter, FillerParameter_VarianceNorm};
use crate::util::math_functions::{caffe_rng_uniform, caffe_rng_gaussian, caffe_rng_bernoulli_i32};


/// Fills a Blob with constant or randomly-generated data.
pub trait Filler<T: BlobType> {
    fn fill(&self, blob: &mut Blob<T>);
}


/// Fills a Blob with constant values $ x = 0 $.
pub struct ConstantFiller {
    filler_param: FillerParameter,
}

impl ConstantFiller {
    pub fn new(param: &FillerParameter) -> Self {
        ConstantFiller {
            filler_param: param.clone()
        }
    }
}

impl<T: BlobType> Filler<T> for ConstantFiller {
    fn fill(&self, blob: &mut Blob<T>) {
        let count = blob.count();
        let data = blob.mutable_cpu_data();
        let value = self.filler_param.get_value();
        debug_assert_eq!(count, data.len());
        assert_ne!(count, 0);

        data.fill(T::from_f32(value));
        assert_eq!(self.filler_param.get_sparse(), -1, "Sparsity not supported by this Filler.");
    }
}


/// Fills a Blob with uniformly distributed values $ x\sim U(a, b) $.
pub struct UniformFiller {
    filler_param: FillerParameter,
}

impl UniformFiller {
    pub fn new(param: &FillerParameter) -> Self {
        UniformFiller {
            filler_param: param.clone()
        }
    }
}

impl<T: BlobType> Filler<T> for UniformFiller {
    fn fill(&self, blob: &mut Blob<T>) {
        let count = blob.count();
        assert_ne!(count, 0);
        caffe_rng_uniform(count, T::from_f32(self.filler_param.get_min()),
                          T::from_f32(self.filler_param.get_max()), blob.mutable_cpu_data());
        assert_eq!(self.filler_param.get_sparse(), -1, "Sparsity not supported by this Filler.");
    }
}


/// Fills a Blob with Gaussian-distributed values $ x = a $.
pub struct GaussianFiller {
    filler_param: FillerParameter,
}

impl GaussianFiller {
    pub fn new(param: &FillerParameter) -> Self {
        GaussianFiller {
            filler_param: param.clone(),
        }
    }
}

impl<T: BlobType> Filler<T> for GaussianFiller {
    fn fill(&self, blob: &mut Blob<T>) {
        let count = blob.count();
        assert_ne!(count, 0);
        caffe_rng_gaussian(count, T::from_f32(self.filler_param.get_mean()),
                           T::from_f32(self.filler_param.get_std()), blob.mutable_cpu_data());
        let sparse = self.filler_param.get_sparse();
        assert!(sparse >= -1);
        if sparse >= 0 {
            // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
            // These have num == channels == 1; width is number of inputs; height is
            // number of outputs.  The 'sparse' variable specifies the mean number
            // of non-zero input weights for a given output.
            assert!(blob.num_axes() >= 1);
            let num_outputs = blob.shape_idx(0);
            let non_zero_probability = T::from_div(T::from_i32(sparse) / T::from_i32(num_outputs));
            let mut rand_vec = Vec::with_capacity(count);
            rand_vec.resize(count, 0);
            caffe_rng_bernoulli_i32(count, non_zero_probability, &mut rand_vec);
            let data = blob.mutable_cpu_data();
            for i in 0..count {
                // SAFETY: Blob data size and `rand_vec` data size both equal to `count`.
                unsafe { *data.get_unchecked_mut(i) *= T::from_i32(*rand_vec.get_unchecked(i)); }
            }
        }
    }
}


/// Fills a Blob with values $ x \in [0, 1] $ such that $ \forall i \sum_j x_{ij} = 1 $.
pub struct PositiveUnitballFiller {
    filler_param: FillerParameter,
}

impl PositiveUnitballFiller {
    pub fn new(param: &FillerParameter) -> Self {
        PositiveUnitballFiller {
            filler_param: param.clone(),
        }
    }
}

impl<T: BlobType> Filler<T> for PositiveUnitballFiller {
    fn fill(&self, blob: &mut Blob<T>) {
        let count = blob.count();
        assert_ne!(count, 0);
        caffe_rng_uniform(count, T::from_i32(0), T::from_i32(1), blob.mutable_cpu_data());
        // We expect the filler to not be called very frequently, so we will
        // just use a simple implementation
        let num = blob.shape_idx(0) as usize;
        let dim = count / num;
        assert_ne!(dim, 0);
        let data = blob.mutable_cpu_data();
        for i in 0..num {
            let mut sum = T::default();
            for j in 0..dim {
                // SAFETY: max value of `i*dim+j` is `count` which is the data size.
                sum += unsafe { *data.get_unchecked(i * dim + j) };
            }
            for j in 0..dim {
                // SAFETY: max value of `i*dim+j` is `count` which is the data size.
                unsafe { *data.get_unchecked_mut(i * dim + j) /= sum; }
            }
        }

        assert_eq!(self.filler_param.get_sparse(), -1, "Sparsity not supported by this Filler.");
    }
}


/// Fills a Blob with values $ x \sim U(-a, +a) $ where $ a $ is set inversely proportional
/// to number of incoming nodes, outgoing nodes, or their average.
///
/// A Filler based on the paper \[Bengio and Glorot 2010\]: Understanding the difficulty
/// of training deep feedforward neuralnetworks.
///
/// It fills the incoming matrix by randomly sampling uniform data from [-scale, scale] where
/// scale = sqrt(3 / n) where n is the fan_in, fan_out, or their average, depending on the
/// variance_norm option. You should make sure the input blob has shape (num, a, b, c) where
/// a * b * c = fan_in and num * b * c = fan_out. Note that this is currently not the case
/// for inner product layers.
pub struct XavierFiller {
    filler_param: FillerParameter,
}

impl XavierFiller {
    pub fn new(param: &FillerParameter) -> Self {
        XavierFiller {
            filler_param: param.clone(),
        }
    }
}

impl<T: BlobType> Filler<T> for XavierFiller {
    fn fill(&self, blob: &mut Blob<T>) {
        let count = blob.count();
        assert_ne!(count, 0);
        let n = get_fan(&self.filler_param, blob);
        let scale = T::sqrt(T::from_div(T::from_i32(3) / n));
        let mut neg_scale = T::default();
        neg_scale -= scale;
        caffe_rng_uniform(count, neg_scale, scale, blob.mutable_cpu_data());
        assert_eq!(self.filler_param.get_sparse(), -1, "Sparsity not supported by this Filler.");
    }
}

// Used in `XavierFiller`, `MSRAFiller`.
fn get_fan<T: BlobType>(param: &FillerParameter, blob: &Blob<T>) -> T {
    let count = blob.count();
    let fan_in = count / blob.shape_idx(0) as usize;
    // Compatibility with ND blobs
    let fan_out = if blob.num_axes() > 1 { count / blob.shape_idx(1) as usize } else { count };
    let variance = param.get_variance_norm();
    if variance == FillerParameter_VarianceNorm::AVERAGE {
        T::from_f64((fan_in + fan_out) as f64 / 2f64)
    } else if variance == FillerParameter_VarianceNorm::FAN_OUT {
        T::from_usize(fan_out)
    } else {
        T::from_usize(fan_in)
    }
}

/// Fills a Blob with values $ x \sim N(0, \sigma^2) $ where $ \sigma^2 $ is set inversely
/// proportional to number of incoming nodes, outgoing nodes, or their average.
///
/// A Filler based on the paper [He, Zhang, Ren and Sun 2015]: Specifically accounts for
/// ReLU nonlinearities.
///
/// Aside: for another perspective on the scaling factor, see the derivation of [Saxe,
/// McClelland, and Ganguli 2013 (v3)].
///
/// It fills the incoming matrix by randomly sampling Gaussian data with std = sqrt(2 / n)
/// where n is the fan_in, fan_out, or their average, depending on the variance_norm option.
/// You should make sure the input blob has shape (num, a, b, c) where a * b * c = fan_in
/// and num * b * c = fan_out. Note that this is currently not the case for inner product layers.
pub struct MSRAFiller {
    filler_param: FillerParameter,
}

impl MSRAFiller {
    pub fn new(param: &FillerParameter) -> Self {
        MSRAFiller {
            filler_param: param.clone()
        }
    }
}

impl<T: BlobType> Filler<T> for MSRAFiller {
    fn fill(&self, blob: &mut Blob<T>) {
        let count = blob.count();
        assert_ne!(count, 0);
        let n = get_fan(&self.filler_param, blob);
        let mut std = T::from_i32(2);
        std /= n;
        std = T::sqrt(std);
        caffe_rng_gaussian(count, T::default(), std, blob.mutable_cpu_data());
        assert_eq!(self.filler_param.get_sparse(), -1, "Sparsity not supported by this Filler.");
    }
}


/// Fills a Blob with coefficients for bilinear interpolation.
///
/// A common use case is with the DeconvolutionLayer acting as upsampling. You can upsample
/// a feature map with shape of (B, C, H, W) by any integer factor using the following proto.
///
/// ``` proto
/// layer {
///   name: "upsample"
///   type: "Deconvolution"
///   bottom: "{{bottom_name}}"
///   top: "{{top_name}}"
///   convolution_param {
///     kernel_size: {{2 * factor - factor % 2}}
///     stride: {{factor}}
///     num_output: {{C}}
///     group: {{C}}
///     pad: {{ceil((factor - 1) / 2.)}}
///     weight_filler: { type: "bilinear" }
///     bias_term: false
///   }
///   param { lr_mult: 0 decay_mult: 0 }
/// }
/// ```
///
/// Please use this by replacing `{{}}` with your values. By specifying
/// `num_output: {{C}} group: {{C}}`, it behaves as channel-wise convolution. The filter
/// shape of this deconvolution layer will be (C, 1, K, K) where K is `kernel_size`, and
/// this filler will set a (K, K) interpolation kernel for every channel of the filter
/// identically. The resulting shape of the top feature map will be (B, C, factor * H, factor * W).
/// Note that the learning rate and the weight decay are set to 0 in order to keep coefficient
/// values of bilinear interpolation unchanged during training. If you apply this to an image,
/// this operation is equivalent to the following call in Python with `Scikit.Image`.
///
/// ``` python
/// out = skimage.transform.rescale(img, factor, mode='constant', cval=0)
/// ```
pub struct BilinearFiller {
    filler_layer: FillerParameter,
}

impl BilinearFiller {
    pub fn new(param: &FillerParameter) -> Self {
        BilinearFiller {
            filler_layer: param.clone()
        }
    }
}

impl<T: BlobType> Filler<T> for BilinearFiller {
    fn fill(&self, blob: &mut Blob<T>) {
        assert_eq!(blob.num_axes(), 4, "Blob must be 4 dim.");
        let width = blob.width() as usize;
        let height = blob.height() as usize;
        assert_eq!(width, height, "Filter must be square");
        let f = (width as f64 / 2f64).ceil();
        let c = T::from_f64((width - 1) as f64 / (2f64 * f));
        let f = T::from_f64(f);
        let count = blob.count();
        let data = blob.mutable_cpu_data();
        for i in 0..count {
            let mut x = T::from_usize(i % width);
            let mut y = T::from_usize((i / width) % height);
            x /= f;
            x -= c;
            let mut xx = T::from_i32(1);
            xx -= T::fabs(x);
            y /= f;
            y -= c;
            let mut yy = T::from_i32(1);
            yy -= T::fabs(y);

            xx *= yy;
            unsafe { *data.get_unchecked_mut(i) = xx; }
        }

        assert_eq!(self.filler_layer.get_sparse(), -1, "Sparsity not supported by this Filler.");
    }
}


/// Get a specific filler from the specification given in FillerParameter.
pub fn get_filler<T: BlobType>(param: &FillerParameter) -> Box<dyn Filler<T>> {
    let ty = param.get_field_type();
    match ty {
        "constant" => Box::new(ConstantFiller::new(param)),
        "gaussian" => Box::new(GaussianFiller::new(param)),
        "positive_unitball" => Box::new(PositiveUnitballFiller::new(param)),
        "uniform" => Box::new(UniformFiller::new(param)),
        "xavier" => Box::new(XavierFiller::new(param)),
        "msra" => Box::new(MSRAFiller::new(param)),
        "bilinear" => Box::new(BilinearFiller::new(param)),
        _ => panic!("Unknown filler name: {:?}", ty),
    }
}
