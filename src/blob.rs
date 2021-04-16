use std::boxed::Box;
use std::rc::Rc;
use std::cell::{RefCell, Ref, RefMut};

use crate::synced_mem::{SyncedMemory, MemShared, MemPtr};
use crate::util::math_functions::{Blas, CaffeUtil};
use crate::proto::caffe;

use std::borrow::Borrow;
use std::option::Option::Some;
use crate::proto::caffe::BlobProto;


pub trait FromFloat: Copy {
    fn from_f64(v: f64) -> Self;

    fn from_f32(v: f32) -> Self;
}

impl FromFloat for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }

    fn from_f32(v: f32) -> Self {
        v
    }
}

impl FromFloat for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }

    fn from_f32(v: f32) -> Self {
        v as f64
    }
}

/// A marker trait to be used in the type bound of `Blob`. It is explicitly marked as `unsafe` and
/// only should be implemented for `f32` and `f64` currently.
pub unsafe trait BlobType: Default + FromFloat {}

unsafe impl BlobType for f32 {}

unsafe impl BlobType for f64 {}

#[derive(Copy, Clone)]
pub struct BlobMemRef<'a, T> {
    pub data: &'a [T],
    pub diff: &'a [T],
}

pub struct BlobMemRefMut<'a, T> {
    pub data: &'a mut [T],
    pub diff: &'a mut [T],
}

/// A wrapper around `SyncedMemory` holders serving as the basic computational unit
/// through which `Layer`, `Net` <strike>and `Solver`</strike> interact.
#[derive(Default)]
pub struct Blob<T: BlobType> {
    data: Option<Rc<RefCell<SyncedMemory<T>>>>,
    diff: Option<Rc<RefCell<SyncedMemory<T>>>>,
    // shape_data: Option<Box<SyncedMemory<T>>>,
    shape: Vec<i32>,
    count: usize,
    capacity: usize,
}

const MAX_BLOB_AXES: i32 = 32;

impl<T> Blob<T> where T: BlobType {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_shape(shape: &Vec<i32>) -> Self {
        let mut blob = Blob::new();
        blob.reshape(shape);
        blob
    }

    #[inline]
    pub fn num_axes(&self) -> i32 {
        self.shape.len() as i32
    }

    #[inline]
    pub fn shape(&self) -> &Vec<i32> {
        &self.shape
    }

    // pub fn gpu_shape(&self) -> &[i32] {}

    #[inline]
    pub fn shape_idx(&self, index: i32) -> i32 {
        self.shape[self.canonical_axis_index(index)]
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Compute the volume of a slice; i.e., the product of dimensions among a range of axes.
    ///
    /// * `start`: The first axis to include in the slice.
    /// * `end`: The first axis to exclude from the slice.
    pub fn count_range(&self, start: usize, end: usize) -> i32 {
        if start == end {
            1
        } else {
            self.shape[start..end].iter().sum()
        }
    }

    /// Compute the volume of a slice spanning from a particular first axis (`start`) to the final axis.
    #[inline]
    pub fn count_range_to_end(&self, start: usize) -> i32 {
        self.count_range(start, self.shape.len())
    }

    pub fn canonical_axis_index(&self, index: i32) -> usize {
        let axes = self.num_axes();
        check_ge!(index, -axes, "axis {:?} out of range for {:?}-D Blob with shape {:?}",
                  index, axes, self.shape_string());
        check_lt!(index, axes, "axis {:?} out of range for {:?}-D Blob with shape {:?}",
                  index, axes, self.shape_string());

        if index < 0 {
            (index + axes) as usize
        } else {
            index as usize
        }
    }

    pub fn shape_string(&self) -> String {
        let capacity = self.shape.len() * 4;
        let mut s = String::with_capacity(capacity);
        for &x in &self.shape {
            s.push_str(x.to_string().as_str());
            s.push(' ');
        }
        s.push('(');
        s.push_str(self.count.to_string().as_str());
        s.push(')');

        s
    }

    pub fn offset(&self, n: i32, c: i32, h: i32, w: i32) -> i32 {
        check_ge!(n, 0);
        check_le!(n, self.shape_idx(0));
        check_ge!(c, 0);
        check_le!(c, self.shape_idx(1));
        check_ge!(h, 0);
        check_le!(h, self.shape_idx(2));
        check_ge!(w, 0);
        check_le!(w, self.shape_idx(3));

        ((n * self.shape_idx(1) + c) * self.shape_idx(2) + h) * self.shape_idx(3) + w
    }

    pub fn offset_idx(&self, indices: &Vec<i32>) -> i32 {
        check_le!(indices.len(), self.shape.len());

        let mut offset = 0;
        let mut idx: usize = 0;
        let len = indices.len();
        for &x in &self.shape {
            offset *= x;
            if len > idx {
                let v = indices[idx];
                check_ge!(v, 0);
                check_le!(v, x);
                offset += v;
            }
        }

        offset
    }

    pub fn reshape(&mut self, shape: &Vec<i32>) {
        check_le!(shape.len() as i32, MAX_BLOB_AXES);

        let mut count = 1;
        for &x in shape {
            check_ge!(x, 0);    // maybe should constrain with x>0?
            if count != 0 {
                check_le!(x, i32::MAX / count, "blob size exceeds INT_MAX");
            }

            count *= x;
        }

        let count = count as usize;
        self.count = count;
        self.shape.clone_from(shape);

        if count > self.capacity {
            self.capacity = count;
            self.data = Some(Rc::new(RefCell::new(SyncedMemory::new(count))));
            self.diff = Some(Rc::new(RefCell::new(SyncedMemory::new(count))));
        }
    }

    pub fn reshape_like(&mut self, other: &Blob<T>) {
        self.reshape(other.shape());
    }

    pub fn reshape_with(&mut self, shape: &caffe::BlobShape) {
        check_le!(shape.get_dim().len() as i32, MAX_BLOB_AXES);

        let mut shape_vec = Vec::with_capacity(shape.get_dim().len());
        for &x in shape.get_dim() {
            shape_vec.push(x as i32);
        }
        self.reshape(&shape_vec);
    }

    pub fn cpu_data(&self) -> &[T] {
        let (ptr, count) = self.data.as_ref().unwrap().borrow_mut().cpu_data().raw_parts();
        unsafe { std::slice::from_raw_parts(ptr, count) }
    }

    pub fn cpu_data_shared(&self) -> MemShared<T> {
        self.data.as_ref().unwrap().borrow_mut().cpu_data_shared()
    }

    // pub fn gpu_data(&mut self) -> &[T] {}

    pub fn cpu_diff(&self) -> &[T] {
        if let Some(ref ptr) = self.diff {
            let (ptr, count) = (*ptr).borrow_mut().cpu_data().raw_parts();
            unsafe { std::slice::from_raw_parts(ptr, count) }
        } else {
            panic!("diff memory not init");
        }
    }

    pub fn cpu_diff_shared(&self) -> MemShared<T> {
        self.diff.as_ref().unwrap().borrow_mut().cpu_data_shared()
    }

    // pub fn gpu_diff(&mut self) -> &[T] {}

    pub fn cpu_mem_ref(&self) -> BlobMemRef<T> {
        let (data_ptr, data_count) = self.data.as_ref().unwrap().borrow_mut().cpu_data().raw_parts();
        let (diff_ptr, diff_count) = self.diff.as_ref().unwrap().borrow_mut().cpu_data().raw_parts();
        BlobMemRef {
            data: unsafe { std::slice::from_raw_parts(data_ptr, data_count) },
            diff: unsafe { std::slice::from_raw_parts(diff_ptr, diff_count) },
        }
    }

    pub fn mutable_cpu_data(&mut self) -> &mut [T] {
        if let Some(ref mut ptr) = self.data {
            let (ptr, count) = (*ptr).borrow_mut().mutable_cpu_data().raw_parts();
            unsafe { std::slice::from_raw_parts_mut(ptr, count) }
        } else {
            panic!("data memory not init");
        }
    }

    pub fn mutable_cpu_diff(&mut self) -> &mut [T] {
        if let Some(ref mut ptr) = self.diff {
            let (ptr, count) = (*ptr).borrow_mut().mutable_cpu_data().raw_parts();
            unsafe { std::slice::from_raw_parts_mut(ptr, count) }
        } else {
            panic!("diff memory not init");
        }
    }

    pub fn mutable_cpu_mem_ref(&mut self) -> BlobMemRefMut<T> {
        let (data_ptr, data_count) = self.data.as_ref().unwrap().borrow_mut().mutable_cpu_data().raw_parts();
        let (diff_ptr, diff_count) = self.diff.as_ref().unwrap().borrow_mut().mutable_cpu_data().raw_parts();
        BlobMemRefMut {
            data: unsafe { std::slice::from_raw_parts_mut(data_ptr, data_count) },
            diff: unsafe { std::slice::from_raw_parts_mut(diff_ptr, diff_count) },
        }
    }

    pub fn set_cpu_data(&mut self, data: &MemShared<T>) {
        if let Some(ref mut ptr) = self.data {
            let data_count = (*ptr).as_ref().borrow().count();
            if data_count != self.count {
                self.data = Some(Rc::new(RefCell::new(SyncedMemory::new(self.count))));
                self.diff = Some(Rc::new(RefCell::new(SyncedMemory::new(self.count))));
            }
            self.data.as_ref().unwrap().borrow_mut().set_cpu_data(data);
        } else {
            panic!("data memory not init");
        }
    }

    pub fn share_data(&mut self, other: &Blob<T>) {
        check_eq!(self.count, other.count());

        if let Some(ref ptr) = other.data {
            self.data = Some(Rc::clone(ptr));
        } else {
            panic!("data memory of other not init");
        }
    }

    #[inline]
    pub fn data_at(&self, n: i32, c: i32, h: i32, w: i32) -> T {
        self.cpu_data()[self.offset(n, c, h, w) as usize]
    }

    #[inline]
    pub fn diff_at(&self, n: i32, c: i32, h: i32, w: i32) -> T {
        self.cpu_diff()[self.offset(n, c, h, w) as usize]
    }

    #[inline]
    pub fn data_at_idx(&self, index: &Vec<i32>) -> T {
        self.cpu_data()[self.offset_idx(index) as usize]
    }

    #[inline]
    pub fn diff_at_idx(&self, index: &Vec<i32>) -> T {
        self.cpu_diff()[self.offset_idx(index) as usize]
    }

    pub fn share_diff(&mut self, other: &Blob<T>) {
        check_eq!(self.count, other.count());

        if let Some(ref ptr) = other.diff {
            self.diff = Some(Rc::clone(ptr));
        } else {
            panic!("diff memory of other not init");
        }
    }

    pub fn copy_from(&mut self, source: &Blob<T>, copy_diff: bool, reshape: bool) {
        if (self.count != source.count) || (self.shape != source.shape) {
            if reshape {
                self.reshape_like(source);
            } else {
                panic!("Trying to copy blobs of different sizes.");
            }
        }

        if copy_diff {
            CaffeUtil::caffe_copy(self.count, source.cpu_diff(), self.mutable_cpu_diff());
        } else {
            CaffeUtil::caffe_copy(self.count, source.cpu_data(), self.mutable_cpu_data());
        }
    }

    // deprecated
    pub fn legacy_shape(&self, index: i32) -> i32 {
        check_le!(self.num_axes(), 4);
        check_lt!(index, 4);
        check_ge!(index, -4);

        if index >= self.num_axes() || index < -self.num_axes() {
            return 1;
        }
        self.shape_idx(index)
    }

    #[inline]
    pub fn num(&self) -> i32 {
        self.legacy_shape(0)
    }

    #[inline]
    pub fn channels(&self) -> i32 {
        self.legacy_shape(1)
    }

    #[inline]
    pub fn height(&self) -> i32 {
        self.legacy_shape(2)
    }

    #[inline]
    pub fn width(&self) -> i32 {
        self.legacy_shape(3)
    }

    pub fn shape_equals(&self, other: &caffe::BlobProto) -> bool {
        if other.has_num() || other.has_channels() || other.has_height() || other.has_width() {
            return self.num_axes() <= 4 &&
                self.legacy_shape(-4) == other.get_num() &&
                self.legacy_shape(-3) == other.get_channels() &&
                self.legacy_shape(-2) == other.get_height() &&
                self.legacy_shape(-1) == other.get_width();
        }

        let other_shape = other.get_shape().get_dim();
        let mut shape_vec = Vec::with_capacity(other_shape.len());
        for &x in other_shape {
            shape_vec.push(x as i32);
        }

        self.shape == shape_vec
    }

    pub fn set_from_proto(&mut self, proto: &BlobProto, reshape: bool) {
        if reshape {
            let mut shape = Vec::new();
            if proto.has_num() || proto.has_channels() || proto.has_height() || proto.has_width() {
                shape.reserve(4);
                shape.push(proto.get_num());
                shape.push(proto.get_channels());
                shape.push(proto.get_height());
                shape.push(proto.get_width());
            } else {
                let other_shape = proto.get_shape().get_dim();
                shape.reserve(other_shape.len());
                for &x in other_shape {
                    shape.push(x as i32);
                }
            }
            self.reshape(&shape);
        } else {
            assert!(self.shape_equals(proto), "shape mismatch (reshape not set)");
        }

        {
            // copy data
            let count = self.count;
            let mut data_vec = self.mutable_cpu_data();
            if !proto.get_double_data().is_empty() {
                let f64_data = proto.get_double_data();
                check_eq!(count, f64_data.len());

                for i in 0..count {
                    data_vec[i] = T::from_f64(f64_data[i]);
                }
            } else {
                let f32_data = proto.get_data();
                check_eq!(count, f32_data.len());

                for i in 0..count {
                    data_vec[i] = T::from_f32(f32_data[i]);
                }
            }
        }
        {
            // check if copy diff
            let count = self.count;
            if !proto.get_double_diff().is_empty() {
                let f64_diff = proto.get_double_diff();
                check_eq!(count, f64_diff.len());

                let mut diff_vec = self.mutable_cpu_diff();
                for i in 0..count {
                    diff_vec[i] = T::from_f64(f64_diff[i]);
                }
            } else if !proto.get_diff().is_empty() {
                let f32_diff = proto.get_diff();
                check_eq!(count, f32_diff.len());

                let mut diff_vec = self.mutable_cpu_diff();
                for i in 0..count {
                    diff_vec[i] = T::from_f32(f32_diff[i]);
                }
            }
        }
    }
}

impl Blob<f32> {
    pub fn update(&mut self) {
        let count = self.count as i32;
        let mem_ref = self.mutable_cpu_mem_ref();
        Blas::<f32>::caffe_axpy(count, -1.0f32, mem_ref.diff, mem_ref.data);
    }

    pub fn asum_data(&self) -> f32 {
        if let Some(ref ptr) = self.data {
            let data = (*ptr).as_ref().borrow();
            data.try_map_cpu_data(|slice| Blas::<f32>::caffe_cpu_asum(self.count as i32, slice))
                .unwrap_or(0.0f32)
        } else {
            0.0f32
        }
    }

    pub fn asum_diff(&self) -> f32 {
        self.diff.as_ref().map_or(0.0f32, |ptr| {
            let data = (*ptr).as_ref().borrow();
            data.try_map_cpu_data(|slice| Blas::<f32>::caffe_cpu_asum(self.count as i32, slice))
                .unwrap_or(0.0f32)
        })
    }

    fn sumsq_cpu(mem: &Option<Rc<RefCell<SyncedMemory<f32>>>>, count: i32) -> f32 {
        mem.as_ref().map_or(0.0f32, |ptr| {
            let data = (*ptr).as_ref().borrow();
            data.try_map_cpu_data(|slice| Blas::<f32>::caffe_cpu_dot(count, slice, slice))
                .unwrap_or(0.0f32)
        })
    }

    pub fn sumsq_data(&self) -> f32 {
        Self::sumsq_cpu(&self.data, self.count as i32)
    }

    pub fn sumsq_diff(&self) -> f32 {
        Self::sumsq_cpu(&self.diff, self.count as i32)
    }

    fn scale_cpu(mem: &Option<Rc<RefCell<SyncedMemory<f32>>>>, count: i32, scale_factor: f32) {
        if let Some(ref ptr) = mem {
            let mut data = (*ptr).borrow_mut();
            data.try_map_cpu_mut_data(|slice| Blas::<f32>::caffe_scal(count, scale_factor, slice));
        }
    }

    pub fn scale_data(&mut self, scale_factor: f32) {
        Self::scale_cpu(&self.data, self.count as i32, scale_factor)
    }

    pub fn scale_diff(&mut self, scale_factor: f32) {
        Self::scale_cpu(&self.diff, self.count as i32, scale_factor)
    }

    pub fn data_to_proto(&self, proto: &mut BlobProto, write_diff: bool) {
        proto.clear_shape();
        {
            let mut shape_dim = Vec::with_capacity(self.shape.len());
            for &i in &self.shape {
                shape_dim.push(i as i64);
            }
            proto.mut_shape().set_dim(shape_dim);
        }

        proto.clear_data();
        proto.clear_diff();
        {
            let data_vec = self.cpu_data();
            for &i in data_vec {
                proto.mut_data().push(i);
            }
        }

        if write_diff {
            let diff_vec = self.cpu_diff();
            for &i in diff_vec {
                proto.mut_diff().push(i);
            }
        }
    }
}

impl Blob<f64> {
    pub fn update(&mut self) {
        let count = self.count as i32;
        let mem_ref = self.mutable_cpu_mem_ref();
        Blas::<f64>::caffe_axpy(count, -1.0f64, mem_ref.diff, mem_ref.data);
    }

    fn asum_cpu(mem: &Option<Rc<RefCell<SyncedMemory<f64>>>>, count: i32) -> f64 {
        mem.as_ref().map_or(0.0f64, |ptr| {
            let data = (*ptr).as_ref().borrow();
            data.try_map_cpu_data(|slice| Blas::<f64>::caffe_cpu_asum(count, slice))
                .unwrap_or(0.0f64)
        })
    }

    pub fn asum_data(&self) -> f64 {
        Self::asum_cpu(&self.data, self.count as i32)
    }

    pub fn asum_diff(&self) -> f64 {
        Self::asum_cpu(&self.diff, self.count as i32)
    }

    fn sumsq_cpu(mem: &Option<Rc<RefCell<SyncedMemory<f64>>>>, count: i32) -> f64 {
        mem.as_ref().map_or(0.0f64, |ptr| {
            let data = (*ptr).as_ref().borrow();
            data.try_map_cpu_data(|slice| Blas::<f64>::caffe_cpu_dot(count, slice, slice))
                .unwrap_or(0.0f64)
        })
    }

    pub fn sumsq_data(&self) -> f64 {
        Self::sumsq_cpu(&self.data, self.count as i32)
    }

    pub fn sumsq_diff(&self) -> f64 {
        Self::sumsq_cpu(&self.diff, self.count as i32)
    }

    fn scale_cpu(mem: &Option<Rc<RefCell<SyncedMemory<f64>>>>, count: i32, scale_factor: f64) {
        if let Some(ref ptr) = mem {
            let mut data = (*ptr).borrow_mut();
            data.try_map_cpu_mut_data(|slice| Blas::<f64>::caffe_scal(count, scale_factor, slice));
        }
    }

    pub fn scale_data(&mut self, scale_factor: f64) {
        Self::scale_cpu(&self.data, self.count as i32, scale_factor)
    }

    pub fn scale_diff(&mut self, scale_factor: f64) {
        Self::scale_cpu(&self.diff, self.count as i32, scale_factor)
    }

    pub fn data_to_proto(&self, proto: &mut BlobProto, write_diff: bool) {
        proto.clear_shape();
        {
            let mut shape = proto.mut_shape();
            for &i in &self.shape {
                shape.mut_dim().push(i as i64);
            }
        }

        proto.clear_double_data();
        proto.clear_double_diff();
        {
            let data_vec = self.cpu_data();
            for &i in data_vec {
                proto.mut_double_data().push(i);
            }
        }

        if write_diff {
            let diff_vec = self.cpu_diff();
            for &i in diff_vec {
                proto.mut_double_diff().push(i);
            }
        }
    }
}
