use std::boxed::Box;
use std::marker::PhantomData;

use crate::synced_mem::SyncedMemory;


/// A marker trait to used in the bound of `Blob`. It is explicitly marked as `unsafe` and
/// only should implement for `f32` and `f64`.
pub unsafe trait BlobType: Sized + Default {}

unsafe impl BlobType for f32 {}

unsafe impl BlobType for f64 {}

/// A wrapper around `SyncedMemory` holders serving as the basic computational unit
/// through which `Layer`, `Net` <strike>and `Solver`</strike> interact.
#[derive(Default)]
pub struct Blob<'a, T: BlobType> {
    data: Option<Box<SyncedMemory<'a>>>,
    diff: Option<Box<SyncedMemory<'static>>>,
    // shape_data: Option<Box<SyncedMemory<'_>>>,
    shape: Vec<i32>,
    count: i32,
    capacity: i32,
    _phantom: PhantomData<T>,
}

const MAX_BLOB_AXES: i32 = 32;

impl<T> Blob<'_, T> where T: BlobType {
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

    #[inline]
    pub fn shape_idx(&self, index: i32) -> i32 {
        self.shape[self.canonical_axis_index(index)]
    }

    #[inline]
    pub fn count(&self) -> i32 {
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

        self.count = count;
        self.shape.clone_from(shape);

        if count > self.capacity {
            self.capacity = count;
            let ty_size = std::mem::size_of::<T>() * count as usize;
            self.data = Some(Box::new(SyncedMemory::new(ty_size)));
            self.diff = Some(Box::new(SyncedMemory::new(ty_size)));
        }
    }

    pub fn reshape_like(&mut self, other: &Blob<'_, T>) {
        self.reshape(other.shape());
    }
}
