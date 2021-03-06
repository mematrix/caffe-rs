use std::any::TypeId;
use std::borrow::Borrow;
use std::boxed::Box;
use std::cell::{RefCell, Ref, RefMut};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use crate::proto::caffe::{BlobProto, BlobShape};
use crate::synced_mem::{SyncedMemory, MemShared, ArcSyncedMemory};
use crate::util::math_functions::{CaffeNum, caffe_copy};


/// A marker trait to be used in the type bound of `Blob`. It is explicitly marked as `unsafe` and
/// only should be implemented for `f32` and `f64` currently (impl for `i32` partially).
pub unsafe trait BlobType: CaffeNum + std::fmt::Debug + 'static {}

unsafe impl BlobType for i32 {}

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


/// A wrapper around [`SyncedMemory`][SyncedMemory] holders serving as the basic computational unit
/// through which `Layer`, `Net` <strike>and `Solver`</strike> interact.
///
/// [SyncedMemory]: caffe_rs::synced_mem::SyncedMemory
#[derive(Default)]
pub struct Blob<T: BlobType> {
    data: Option<Rc<RefCell<SyncedMemory<T>>>>,
    diff: Option<Rc<RefCell<SyncedMemory<T>>>>,
    // shape_data: Option<Box<SyncedMemory<T>>>,
    shape: Vec<i32>,
    count: usize,
    capacity: usize,
}

pub const MAX_BLOB_AXES: i32 = 32;

impl<T> Blob<T> where T: BlobType {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_shape<Q: Borrow<[i32]> + ?Sized>(shape: &Q) -> Self {
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

    pub fn reshape<Q: Borrow<[i32]> + ?Sized>(&mut self, shape: &Q) {
        let shape = shape.borrow();
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
        self.shape = shape.to_vec();

        if count > self.capacity {
            self.capacity = count;
            self.data = Some(Rc::new(RefCell::new(SyncedMemory::new(count))));
            self.diff = Some(Rc::new(RefCell::new(SyncedMemory::new(count))));
        }
    }

    pub fn reshape_like(&mut self, other: &Blob<T>) {
        self.reshape(other.shape());
    }

    pub fn reshape_with(&mut self, shape: &BlobShape) {
        check_le!(shape.get_dim().len() as i32, MAX_BLOB_AXES);

        let mut shape_vec = Vec::with_capacity(shape.get_dim().len());
        for &x in shape.get_dim() {
            shape_vec.push(x as i32);
        }
        self.reshape(&shape_vec);
    }

    pub fn cpu_data(&self) -> &[T] {
        let (ptr, count) = self.data.as_ref().unwrap().borrow_mut().cpu_data_raw();
        unsafe { std::slice::from_raw_parts(ptr, count) }
    }

    pub fn cpu_data_shared(&self) -> MemShared<T> {
        self.data.as_ref().unwrap().borrow_mut().cpu_data_shared()
    }

    // pub fn gpu_data(&mut self) -> &[T] {}

    pub fn cpu_diff(&self) -> &[T] {
        if let Some(ref ptr) = self.diff {
            let (ptr, count) = (*ptr).borrow_mut().cpu_data_raw();
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
        let (data_ptr, data_count) = self.data.as_ref().unwrap().borrow_mut().cpu_data_raw();
        let (diff_ptr, diff_count) = self.diff.as_ref().unwrap().borrow_mut().cpu_data_raw();
        BlobMemRef {
            data: unsafe { std::slice::from_raw_parts(data_ptr, data_count) },
            diff: unsafe { std::slice::from_raw_parts(diff_ptr, diff_count) },
        }
    }

    pub fn mutable_cpu_data(&mut self) -> &mut [T] {
        if let Some(ref mut ptr) = self.data {
            let (ptr, count) = (*ptr).borrow_mut().mutable_cpu_data_raw();
            unsafe { std::slice::from_raw_parts_mut(ptr, count) }
        } else {
            panic!("data memory not init");
        }
    }

    pub fn mutable_cpu_diff(&mut self) -> &mut [T] {
        if let Some(ref mut ptr) = self.diff {
            let (ptr, count) = (*ptr).borrow_mut().mutable_cpu_data_raw();
            unsafe { std::slice::from_raw_parts_mut(ptr, count) }
        } else {
            panic!("diff memory not init");
        }
    }

    pub fn mutable_cpu_mem_ref(&mut self) -> BlobMemRefMut<T> {
        let (data_ptr, data_count) = self.data.as_ref().unwrap().borrow_mut().mutable_cpu_data_raw();
        let (diff_ptr, diff_count) = self.diff.as_ref().unwrap().borrow_mut().mutable_cpu_data_raw();
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
            caffe_copy(self.count, source.cpu_diff(), self.mutable_cpu_diff());
        } else {
            caffe_copy(self.count, source.cpu_data(), self.mutable_cpu_data());
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

    pub fn shape_equals(&self, other: &BlobProto) -> bool {
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

    pub fn update(&mut self) {
        let count = self.count as i32;
        let mem_ref = self.mutable_cpu_mem_ref();
        T::caffe_axpy(count, T::from_f32(-1.0f32), mem_ref.diff, mem_ref.data);
    }

    fn asum_cpu(mem: &Option<Rc<RefCell<SyncedMemory<T>>>>, count: i32) -> T {
        mem.as_ref().map_or(T::default(), |ptr| {
            let data = (*ptr).as_ref().borrow();
            data.try_map_cpu_data(|slice| T::caffe_cpu_asum(count, slice))
                .unwrap_or_default()
        })
    }

    pub fn asum_data(&self) -> T {
        Self::asum_cpu(&self.data, self.count as i32)
    }

    pub fn asum_diff(&self) -> T {
        Self::asum_cpu(&self.diff, self.count as i32)
    }

    fn scale_cpu(mem: &Option<Rc<RefCell<SyncedMemory<T>>>>, count: i32, scale_factor: T) {
        if let Some(ref ptr) = mem {
            let mut data = (*ptr).borrow_mut();
            data.try_map_cpu_mut_data(|slice| T::caffe_scal(count, scale_factor, slice));
        }
    }

    pub fn scale_data(&mut self, scale_factor: T) {
        Self::scale_cpu(&self.data, self.count as i32, scale_factor);
    }

    pub fn scale_diff(&mut self, scale_factor: T) {
        Self::scale_cpu(&self.diff, self.count as i32, scale_factor);
    }

    fn sumsq_cpu(mem: &Option<Rc<RefCell<SyncedMemory<T>>>>, count: i32) -> T {
        mem.as_ref().map_or(T::default(), |ptr| {
            let data = (*ptr).as_ref().borrow();
            data.try_map_cpu_data(|slice| T::caffe_cpu_dot(count, slice, slice))
                .unwrap_or_default()
        })
    }

    pub fn sumsq_data(&self) -> T {
        Self::sumsq_cpu(&self.data, self.count as i32)
    }

    pub fn sumsq_diff(&self) -> T {
        Self::sumsq_cpu(&self.diff, self.count as i32)
    }

    pub fn to_proto(&self, proto: &mut BlobProto, write_diff: bool) {
        proto.clear_shape();
        {
            let mut shape_dim = Vec::with_capacity(self.shape.len());
            for &i in &self.shape {
                shape_dim.push(i as i64);
            }
            proto.mut_shape().set_dim(shape_dim);
        }

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            proto_write_f32_data(proto, self.cpu_data());
            if write_diff {
                proto_write_f32_diff(proto, self.cpu_diff());
            }
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            proto_write_f64_data(proto, self.cpu_data());
            if write_diff {
                proto_write_f64_diff(proto, self.cpu_diff());
            }
        }
    }
}

fn proto_write_f32_data<T: BlobType>(proto: &mut BlobProto, data: &[T]) {
    proto.clear_data();
    proto.clear_diff();
    {
        for &i in data {
            proto.mut_data().push(i.to_f32());
        }
    }
}

fn proto_write_f32_diff<T: BlobType>(proto: &mut BlobProto, diff: &[T]) {
    for &i in diff {
        proto.mut_diff().push(i.to_f32());
    }
}

fn proto_write_f64_data<T: BlobType>(proto: &mut BlobProto, data: &[T]) {
    proto.clear_double_data();
    proto.clear_double_diff();
    {
        for &i in data {
            proto.mut_double_data().push(i.to_f64());
        }
    }
}

fn proto_write_f64_diff<T: BlobType>(proto: &mut BlobProto, diff: &[T]) {
    for &i in diff {
        proto.mut_double_diff().push(i.to_f64());
    }
}




/// A thread-safe version of [`Blob`][Blob].
///
/// [Blob]: caffe_rs::blob::Blob
#[derive(Default)]
pub struct ArcBlob<T: BlobType> {
    data: Option<Arc<Mutex<ArcSyncedMemory<T>>>>,
    diff: Option<Arc<Mutex<ArcSyncedMemory<T>>>>,
    shape: Vec<i32>,
    count: usize,
    capacity: usize,
}

impl<T: BlobType> ArcBlob<T> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn from(mut blob: Blob<T>) -> Result<Self, Blob<T>> {
        let data = blob.data.map(|d| Rc::try_unwrap(d));
        let data = data.map(
            |r| r.map(|rc| ArcSyncedMemory::from(rc.into_inner()))
        );
        let diff = blob.diff.map(|d| Rc::try_unwrap(d));
        let diff = diff.map(
            |r| r.map(|rc| ArcSyncedMemory::from(rc.into_inner()))
        );
        if data.as_ref().map_or(false, |r| r.is_err()) ||
            diff.as_ref().map_or(false, |r| r.is_err()) {
            blob.data = data.map(|r| r.err().unwrap());
            blob.diff = diff.map(|r| r.err().unwrap());
            return Result::Err(blob);
        }

        let data = data.map(|r| r.ok().unwrap());
        let diff = diff.map(|r| r.ok().unwrap());
        if data.as_ref().map_or(false, |r| r.is_err()) ||
            diff.as_ref().map_or(false, |r| r.is_err()) {
            blob.data = data.map(|r| Rc::new(RefCell::new(r.err().unwrap())));
            blob.diff = diff.map(|r| Rc::new(RefCell::new(r.err().unwrap())));
            return Result::Err(blob);
        }

        let data = data.map(
            |d| Arc::new(Mutex::new(d.ok().unwrap()))
        );
        let diff = diff.map(
            |d| Arc::new(Mutex::new(d.ok().unwrap()))
        );
        let arc_blob = ArcBlob {
            data,
            diff,
            shape: blob.shape,
            count: blob.count,
            capacity: blob.capacity,
        };
        Result::Ok(arc_blob)
    }

    pub fn into_blob(mut self) -> Result<Blob<T>, ArcBlob<T>> {
        let data = self.data.map(|a| Arc::try_unwrap(a));
        let data = data.map(
            |r| r.map(|m| m.into_inner().unwrap().into_mem())
        );
        let diff = self.diff.map(|a| Arc::try_unwrap(a));
        let diff = diff.map(
            |r| r.map(|m| m.into_inner().unwrap().into_mem())
        );
        if data.as_ref().map_or(false, |r| r.is_err()) ||
            diff.as_ref().map_or(false, |r| r.is_err()) {
            self.data = data.map(|r| r.err().unwrap());
            self.diff = diff.map(|r| r.err().unwrap());
            return Result::Err(self);
        }

        let data = data.map(|r| r.ok().unwrap());
        let diff = diff.map(|r| r.ok().unwrap());
        if data.as_ref().map_or(false, |r| r.is_err()) ||
            diff.as_ref().map_or(false, |r| r.is_err()) {
            self.data = data.map(|r| Arc::new(Mutex::new(r.err().unwrap())));
            self.diff = diff.map(|r| Arc::new(Mutex::new(r.err().unwrap())));
            return Result::Err(self);
        }

        let data = data.map(
            |d| Rc::new(RefCell::new(d.ok().unwrap()))
        );
        let diff = diff.map(
            |d| Rc::new(RefCell::new(d.ok().unwrap()))
        );
        let blob = Blob {
            data,
            diff,
            shape: self.shape,
            count: self.count,
            capacity: self.capacity,
        };
        Result::Ok(blob)
    }
}

