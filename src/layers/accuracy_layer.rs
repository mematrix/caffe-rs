use crate::blob::{BlobType, Blob};
use crate::layer::{CaffeLayer, LayerImpl, BlobVec};
use crate::proto::caffe::LayerParameter;
use crate::util::math_functions::caffe_set;


/// Computes the classification accuracy for a one-of-many classification task.
pub struct AccuracyLayer<T: BlobType> {
    layer: LayerImpl<T>,
    label_axis: i32,
    outer_num: i32,
    inner_num: i32,
    top_k: u32,
    /// Whether to ignore instances with a certain label.
    has_ignore_label: bool,
    /// The label indicating that an instance should be ignored.
    ignore_label: i32,
    /// Keeps counts of the number of samples per class.
    nums_buffer: Blob<T>,
}

impl<T: BlobType> AccuracyLayer<T> {
    /// `param` provides **AccuracyParameter** accuracy_param, with **AccuracyLayer options**:
    /// - top_k (**optional, default `1`**). Sets the maximum rank $ k $ at which a prediction
    /// is considered correct. For example, if $ k = 5 $, a prediction is counted correct if
    /// the correct label is among the top 5 predicted labels.
    pub fn new(param: &LayerParameter) -> Self {
        AccuracyLayer {
            layer: LayerImpl::new(param),
            label_axis: 0,
            outer_num: 0,
            inner_num: 0,
            top_k: 0,
            has_ignore_label: false,
            ignore_label: 0,
            nums_buffer: Blob::new()
        }
    }
}

impl<T: BlobType> CaffeLayer for AccuracyLayer<T> {
    type DataType = T;

    fn get_impl(&self) -> &LayerImpl<T> {
        &self.layer
    }

    fn get_impl_mut(&mut self) -> &mut LayerImpl<T> {
        &mut self.layer
    }

    fn layer_setup(&mut self, _bottom: &BlobVec<T>, _top: &BlobVec<T>) {
        self.top_k = self.layer.layer_param.get_accuracy_param().get_top_k();
        self.has_ignore_label = self.layer.layer_param.get_accuracy_param().has_ignore_label();
        if self.has_ignore_label {
            self.ignore_label = self.layer.layer_param.get_accuracy_param().get_ignore_label();
        }
    }

    fn reshape(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        let b0 = bottom[0].as_ref().borrow();
        let b1 = bottom[1].as_ref().borrow();
        check_le!(self.top_k as usize, b0.count() / b1.count(),
            "top_k must be less than or equal to the number of classes.");

        let axis = b0.canonical_axis_index(self.layer.layer_param.get_accuracy_param().get_axis());
        self.label_axis = axis as i32;
        self.outer_num = b0.count_range(0, axis);
        self.inner_num = b0.count_range_to_end(axis + 1);
        assert_eq!(self.outer_num * self.inner_num, b1.count() as i32,
                   "Number of labels must match number of predictions; e.g., if label axis == 1 \
                   and prediction shape is (N, C, H, W), label count (number of labels) must be \
                   N*H*W, with integer values in {{0, 1, ..., C-1}}.");

        // Accuracy is a scalar; 0 axes.
        top[0].borrow_mut().reshape(&Vec::new());
        if top.len() > 1 {
            // Per-class accuracy is a vector; 1 axes.
            let top_shape_per_class = vec![b0.shape_idx(self.label_axis)];
            top[1].borrow_mut().reshape(&top_shape_per_class);
            self.nums_buffer.reshape(&top_shape_per_class);
        }
    }

    fn layer_type(&self) -> &'static str {
        "Accuracy"
    }

    fn exact_num_bottom_blobs(&self) -> i32 {
        2
    }

    fn min_top_blobs(&self) -> i32 {
        1
    }

    /// If there are two top blobs, then the second blob will contain accuracies per class.
    fn max_top_blobs(&self) -> i32 {
        2
    }

    /// *Params:*
    ///
    /// `bottom` input Blob vector (length 2):
    /// - $ (N \times C \times H \times W) $, the predictions $ x $, a Blob with values in
    /// $ [-\infty, +\infty] $ indicating the predicted score for each of the $ K = CHW $
    /// classes, Each $ x_n $ is mapped to a predicted label $ \hat{l}_n $ given by its
    /// maximal index: $ \hat{l}\_n = \arg\max\limits\_k x\_{nk} $.
    /// - $ (N \times 1 \times 1 \times 1) $, the labels $ l $, an integer-valued Blob with values
    /// $ l_n \in [0, 1, 2, ..., K - 1] $ indicating the correct class label among the $ K $ classes.
    ///
    /// `top` output Blob vector (length 1):
    /// - $ (1 \times 1 \times 1 \times 1) $, the computed accuracy: $$
    /// \frac{1}{N} \sum\limits_{n=1}^N \delta\{ \hat{l}_n = l_n \}
    /// $$, where $$
    /// \delta\\{\mathrm{condition}\\} = \left\\{
    ///    \begin{array}{lr}
    ///       1 & \mathrm{if condition} \\\\
    ///       0 & \mathrm{otherwise}
    ///    \end{array} \right.
    /// $$.
    fn forward_cpu(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        let mut accuracy = T::default();
        let b0 = bottom[0].as_ref().borrow();
        let b1 = bottom[1].as_ref().borrow();
        let bottom_data = b0.cpu_data();
        let bottom_label = b1.cpu_data();
        let dim = b0.count() as i32 / self.outer_num;
        let num_labels = b0.shape_idx(self.label_axis);
        if top.len() > 1 {
            caffe_set(self.nums_buffer.count(), T::default(), self.nums_buffer.mutable_cpu_data());
            let mut t1 = top[1].borrow_mut();
            let count = t1.count();
            caffe_set(count, T::default(), t1.mutable_cpu_data());
        }

        let mut count = 0;
        for i in 0..self.outer_num {
            for j in 0..self.inner_num {
                let label_value = bottom_label[(i * self.inner_num + j) as usize].to_i32();
                if self.has_ignore_label && self.ignore_label == label_value {
                    continue;
                }

                check_ge!(label_value, 0);
                check_lt!(label_value, num_labels);
                if top.len() > 1 {
                    self.nums_buffer.mutable_cpu_data()[label_value as usize] += T::from_i32(1);
                }

                let prob_of_true_class = bottom_data[(i * dim + label_value * self.inner_num + j) as usize];
                let mut num_better_predictions = -1;    // true_class also counts as "better"
                let top_k = self.top_k as i32;
                // Top-k accuracy
                let mut k = 0;
                while k < num_labels && num_better_predictions < top_k {
                    let v = bottom_data[(i * dim + k * self.inner_num + j) as usize];
                    num_better_predictions += (v >= prob_of_true_class) as i32;
                    k += 1;
                }
                // Check if there are less than top_k predictions
                if num_better_predictions < top_k {
                    accuracy += T::from_i32(1);
                    if top.len() > 1 {
                        let mut t1 = top[1].borrow_mut();
                        t1.mutable_cpu_data()[label_value as usize] += T::from_i32(1);
                    }
                }

                count += 1;
            }
        }

        top[0].borrow_mut().mutable_cpu_data()[0] = if count == 0 {
            T::default()
        } else {
            accuracy / T::from_i32(count)
        };
        if top.len() > 1 {
            let mut t1 = top[1].borrow_mut();
            for i in 0..t1.count() {
                let num = self.nums_buffer.cpu_data()[i];
                let v = if num.is_zero() {
                    T::default()
                } else {
                    t1.cpu_data()[i] / num
                };
                t1.mutable_cpu_data()[i] = v;
            }
        }
        // Accuracy layer should not be used as a loss function.
    }

    fn forward_gpu(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        no_gpu!();
    }

    /// Not implemented -- AccuracyLayer cannot be used as a loss.
    fn backward_cpu(&mut self, _top: &BlobVec<T>, propagate_down: &Vec<bool>, _bottom: &BlobVec<T>) {
        for &prop_down in propagate_down {
            if prop_down {
                unimplemented!();
            }
        }
    }

    fn backward_gpu(&mut self, top: &BlobVec<T>, propagate_down: &Vec<bool>, bottom: &BlobVec<T>) {
        no_gpu!();
    }
}

register_layer_class!(Accuracy);
