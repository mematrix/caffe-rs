use crate::common::{CaffeRng, Caffe};
use crate::blob::{BlobType, Blob};
use crate::proto::caffe::{TransformationParameter, Phase, BlobProto, Datum};
use crate::util::io::read_proto_from_binary_file_or_die;
use crate::util::rng::caffe_rng_rand;


pub struct DataTransformer<T: BlobType> {
    param: TransformationParameter,
    rng: Option<CaffeRng>,
    phase: Phase,
    data_mean: Blob<T>,
    mean_values: Vec<T>
}

impl<T: BlobType> DataTransformer<T> {
    pub fn new(param: &TransformationParameter, phase: Phase) -> Self {
        let mut this = DataTransformer {
            param: param.clone(),
            rng: None,
            phase,
            data_mean: Default::default(),
            mean_values: Default::default(),
        };

        // Check if we want to use mean_file
        if param.has_mean_file() {
            assert!(param.get_mean_value().is_empty(), "Cannot specify mean_file and mean_value at the same time");
            let mean_file = param.get_mean_file();
            if Caffe::root_solver() {
                info!("Loading mean file from: {:?}", mean_file);
            }

            let mut blob_proto = BlobProto::new();
            read_proto_from_binary_file_or_die(mean_file, &mut blob_proto);
            this.data_mean.set_from_proto(&blob_proto, true);
        }
        // Check if we want to use mean_value
        if !param.get_mean_value().is_empty() {
            assert!(!param.has_mean_file(), "Cannot specify mean_file and mean_value at the same time");
            let mean_value = param.get_mean_value();
            this.mean_values.reserve(mean_value.len());
            for &c in mean_value {
                this.mean_values.push(T::from_f32(c));
            }
        }

        this
    }

    pub fn init_rand(&mut self) {
        let needs_rand = self.param.get_mirror() ||
            (self.phase == Phase::TRAIN && self.param.get_crop_size() != 0);
        if needs_rand {
            let rng_seed = caffe_rng_rand();
            self.rng = Some(CaffeRng::new_with_seed(rng_seed as u64));
        } else {
            self.rng = None;
        }
    }

    pub fn transform_datum(&mut self, datum: &Datum, transformed_data: &mut [T]) {
        let data = datum.get_data();
        let datum_channels = datum.get_channels();
        let datum_height = datum.get_height();
        let datum_width = datum.get_width();

        let crop_size = self.param.get_crop_size() as i32;
        let scale = T::from_f32(self.param.get_scale());
        let do_mirror = self.param.get_mirror() && self.rand(2) != 0;
        let has_mean_file = self.param.has_mean_file();
        let has_uint8 = !data.is_empty();
        let has_mean_values = !self.mean_values.is_empty();

        check_gt!(datum_channels, 0);
        check_ge!(datum_height, crop_size);
        check_ge!(datum_width, crop_size);

        if has_mean_file {
            assert_eq!(datum_channels, self.data_mean.channels());
            assert_eq!(datum_height, self.data_mean.height());
            assert_eq!(datum_width, self.data_mean.width());
        }
        if has_mean_values {
            assert!(self.mean_values.len() == 1 || self.mean_values.len() == datum_channels as usize,
                    "Specify either 1 mean_value or as many as channels: {:?}", datum_channels);
            if datum_channels > 1 && self.mean_values.len() == 1 {
                // Replicate the mean_value for simplicity
                let v = self.mean_values[0];
                for _c in 1..datum_channels {
                    self.mean_values.push(v);
                }
            }
        }

        let mut height = datum_height;
        let mut width = datum_width;
        let mut h_off = 0;
        let mut w_off = 0;
        if crop_size != 0 {
            height = crop_size;
            width = crop_size;
            // We only do random crop when we do training.
            if self.phase == Phase::TRAIN {
                h_off = self.rand(datum_height - crop_size + 1);
                w_off = self.rand(datum_width - crop_size + 1);
            } else {
                h_off = (datum_height - crop_size) / 2;
                w_off = (datum_width - crop_size) / 2;
            }
        }

        let mut mean = None;
        if has_mean_file {
            mean = Some(self.data_mean.cpu_data());
        }
        for c in 0..datum_channels {
            for h in 0..height {
                for w in 0..width {
                    let data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
                    let top_index = if do_mirror {
                        (c * height + h) * width + (width - 1 - w)
                    } else {
                        (c * height + h) * width + w
                    };
                    let data_index = data_index as usize;
                    let top_index = top_index as usize;
                    let mut datum_element = if has_uint8 {
                        T::from_i32(data[data_index] as i32)
                    } else {
                        T::from_f32(datum.get_float_data()[data_index])
                    };

                    if has_mean_file {
                        datum_element -= mean.unwrap()[data_index];
                        datum_element *= scale;
                        transformed_data[top_index] = datum_element;
                    } else {
                        if has_mean_values {
                            datum_element -= self.mean_values[c as usize];
                            datum_element *= scale;
                            transformed_data[top_index] = datum_element;
                        } else {
                            datum_element *= scale;
                            transformed_data[top_index] = datum_element;
                        }
                    }
                }
            }
        }
    }

    pub fn transform_datum_blob(&mut self, datum: &Datum, transformed_blob: &mut Blob<T>) {
        // If datum is encoded, decode and transform the cv::image.
        if datum.get_encoded() {
            todo!("OpenCV");
            assert!(false, "Encoded datum requires OpenCV");
        } else {
            if self.param.get_force_color() || self.param.get_force_gray() {
                error!("force_color and force_gray only for encoded datum");
            }
        }

        let crop_size = self.param.get_crop_size() as i32;
        let datum_channels = datum.get_channels();
        let datum_height = datum.get_height();
        let datum_width = datum.get_width();

        // Check dimensions.
        let channels = transformed_blob.channels();
        let height = transformed_blob.height();
        let width = transformed_blob.width();
        let num = transformed_blob.num();

        assert_eq!(channels, datum_channels);
        check_le!(height, datum_height);
        check_le!(width, datum_width);
        check_ge!(num, 1);

        if crop_size != 0 {
            assert_eq!(crop_size, height);
            assert_eq!(crop_size, width);
        } else {
            assert_eq!(datum_height, height);
            assert_eq!(datum_width, width);
        }

        let transformed_data = transformed_blob.mutable_cpu_data();
        self.transform_datum(datum, transformed_data);
    }

    pub fn transform_datum_vec(&mut self, datum_vector: &Vec<Datum>, transformed_blob: &mut Blob<T>) {
        let datum_num = datum_vector.len();
        let num = transformed_blob.num();
        let channels = transformed_blob.channels();
        let height = transformed_blob.height();
        let width = transformed_blob.width();

        assert!(datum_num > 0, "There is no datum to add");
        assert!(datum_num <= num as usize, "The size of datum_vector must be no greater than transformed_blob->num()");
        let shape = vec![1, channels, height, width];
        let mut uni_blob = Blob::with_shape(&shape);
        for item_id in 0..datum_num {
            let offset = transformed_blob.offset(item_id as i32, 0, 0, 0);
            let data = transformed_blob.cpu_data_shared().offset(offset);
            uni_blob.set_cpu_data(&data);
            self.transform_datum_blob(&datum_vector[item_id], &mut uni_blob);
        }
    }

    pub fn transform_blob(&mut self, input_blob: &mut Blob<T>, transformed_blob: &mut Blob<T>) {
        let crop_size = self.param.get_crop_size() as i32;
        let input_num = input_blob.num();
        let input_channels = input_blob.channels();
        let input_height = input_blob.height();
        let input_width = input_blob.width();

        if transformed_blob.count() == 0 {
            // Initialize transformed_blob with the right shape.
            if crop_size != 0 {
                let shape = vec![input_num, input_channels, crop_size, crop_size];
                transformed_blob.reshape(&shape);
            } else {
                let shape = vec![input_num, input_channels, input_height, input_width];
                transformed_blob.reshape(&shape);
            }
        }

        let num = transformed_blob.num();
        let channels = transformed_blob.channels();
        let height = transformed_blob.height();
        let width = transformed_blob.width();
        let size = transformed_blob.count();

        check_le!(input_num, num);
        assert_eq!(input_channels, channels);
        check_ge!(input_height, height);
        check_ge!(input_width, width);

        let scale = self.param.get_scale();
        let do_mirror = self.param.get_mirror() && self.rand(2) != 0;
        let has_mean_file = self.param.has_mean_file();
        let has_mean_values = !self.mean_values.is_empty();

        let mut h_off = 0;
        let mut w_off = 0;
        if crop_size != 0 {
            assert_eq!(crop_size, height);
            assert_eq!(crop_size, width);
            // We only do random crop when we do training.
            let height_diff = input_height - crop_size;
            if self.phase == Phase::TRAIN {
                h_off = self.rand(height_diff + 1);
                w_off = self.rand(input_width - crop_size + 1);
            } else {
                h_off = (height_diff) / 2;
                w_off = (input_width - crop_size) / 2;
            };
        } else {
            assert_eq!(input_height, height);
            assert_eq!(input_width, width);
        }

        // SAFETY: mutable slice data borrowed partial which is not accessed in later `input_blob`
        // immutable read.
        let input_data = unsafe {
            let data = input_blob.mutable_cpu_data();
            std::slice::from_raw_parts_mut(data.as_mut_ptr(), data.len())
        };
        if has_mean_file {
            assert_eq!(input_channels, self.data_mean.channels());
            assert_eq!(input_height, self.data_mean.height());
            assert_eq!(input_width, self.data_mean.width());
            for n in 0..input_num {
                let offset = input_blob.offset(n, 0, 0, 0);
                T::caffe_sub_assign(self.data_mean.count(), &mut input_data[offset as usize..],
                                    self.data_mean.cpu_data());
            }
        }

        if has_mean_values {
            assert!(self.mean_values.len() == 1 || self.mean_values.len() == input_channels as usize,
                    "Specify either 1 mean_value or as many as channels: {:?}", input_channels);
            if self.mean_values.len() == 1 {
                let mut alpha = T::default();
                alpha -= self.mean_values[0];
                T::caffe_add_scalar(input_blob.count(), alpha, input_data);
            } else {
                for n in 0..input_num {
                    for c in 0..input_channels {
                        let offset = input_blob.offset(n, c, 0, 0);
                        let count = (input_height * input_width) as usize;
                        let mut alpha = T::default();
                        alpha -= self.mean_values[c as usize];
                        T::caffe_add_scalar(count, alpha, &mut input_data[offset as usize..]);
                    }
                }
            }
        }

        let transformed_data = transformed_blob.mutable_cpu_data();
        for n in 0..input_num {
            let top_index_n = n * channels;
            let data_index_n = n * channels;
            for c in 0..channels {
                let top_index_c = (top_index_n + c) * height;
                let data_index_c = (data_index_n + c) * input_height + h_off;
                for h in 0..height {
                    let top_index_h = (top_index_c + h) * width;
                    let data_index_h = (data_index_c + h) * input_width + w_off;
                    if do_mirror {
                        let top_index_w = top_index_h + width - 1;
                        for w in 0..width {
                            transformed_data[(top_index_w - w) as usize] = input_data[(data_index_h + w) as usize];
                        }
                    } else {
                        for w in 0..width {
                            transformed_data[(top_index_h + w) as usize] = input_data[(data_index_h + w) as usize];
                        }
                    }
                }
            }
        }
        if scale != 1f32 {
            info!("Scale: {}", scale);
            T::caffe_scal(size as i32, T::from_f32(scale), transformed_data);
        }
    }

    pub fn infer_blob_shape(&self, datum: &Datum) -> Vec<i32> {
        if datum.get_encoded() {
            assert!(false, "Encoded datum requires OpenCV; compile with USE_OPENCV.");
        }

        let crop_size = self.param.get_crop_size() as i32;
        let datum_channels = datum.get_channels();
        let datum_height = datum.get_height();
        let datum_width = datum.get_width();
        // Check dimensions.
        check_gt!(datum_channels, 0);
        check_ge!(datum_height, crop_size);
        check_ge!(datum_width, crop_size);
        // Build BlobShape.
        let height = if crop_size != 0 { crop_size } else { datum_height };
        let width = if crop_size != 0 { crop_size } else { datum_width };
        vec![1, datum_channels, height, width]
    }

    pub fn infer_blob_shape_vec(&self, datum_vector: &Vec<Datum>) -> Vec<i32> {
        let num = datum_vector.len();
        check_gt!(num, 0, "There is no datum to in the vector");
        // Use first datum in the vector to InferBlobShape.
        let mut shape = self.infer_blob_shape(datum_vector.first().unwrap());
        // Adjust num to the size of the vector.
        shape[0] = num as i32;
        shape
    }

    fn rand(&mut self, n: i32) -> i32 {
        assert!(self.rng.is_some());
        assert!(n > 0);
        let r = self.rng.as_mut().unwrap().generator().next_u32() as i32;
        r % n
    }
}
