use std::boxed::Box;
use std::rc::Rc;
use std::cell::{RefCell, Ref, RefMut};

use crate::synced_mem::{SyncedMemory, MemShared, MemPtr};
use crate::util::math_functions::Blas;

use std::borrow::Borrow;
use std::option::Option::Some;


/// A marker trait to be used in the type bound of `Blob`. It is explicitly marked as `unsafe` and
/// only should be implemented for `f32` and `f64` currently.
pub unsafe trait BlobType: Sized + Default + Copy {}

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
    shape_data: Option<Box<SyncedMemory<T>>>,
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
        //
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
}

impl Blob<f64> {
    pub fn update(&mut self) {
        let count = self.count as i32;
        let mem_ref = self.mutable_cpu_mem_ref();
        Blas::<f64>::caffe_axpy(count, -1.0f64, mem_ref.diff, mem_ref.data);
    }

    pub fn asum_data(&self) -> f64 {
        self.data.as_ref().map_or(0.0f64, |ptr| {
            let data = (*ptr).as_ref().borrow();
            data.try_map_cpu_data(|slice| Blas::<f64>::caffe_cpu_asum(self.count as i32, slice))
                .unwrap_or(0.0f64)
        })
    }
}
