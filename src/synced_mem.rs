use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cell::{RefCell, Ref, RefMut};
use std::rc::Rc;
use std::sync::{Arc, Mutex};


struct MemPtr<T> {
    ptr: *mut T,
    count: usize,
}

/// Makes the raw memory handle be sent between threads.
unsafe impl<T> Send for MemPtr<T> {}

impl<T> MemPtr<T> {
    pub fn new(count: usize) -> Self {
        let layout = Layout::array::<T>(count).unwrap();
        trace!("alloc owned memory, layout: {:?}", layout);
        MemPtr {
            ptr: unsafe { alloc_zeroed(layout) as *mut T },
            count,
        }
    }

    pub fn raw_parts(&self) -> (*const T, usize) {
        (self.ptr as *const T, self.count)
    }

    pub fn raw_parts_mut(&mut self) -> (*mut T, usize) {
        (self.ptr, self.count)
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.count) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.count) }
    }
}

impl<T> Drop for MemPtr<T> {
    fn drop(&mut self) {
        let layout = Layout::array::<T>(self.count).unwrap();
        trace!("dealloc owned memory, ptr: {:?}, size: {} * {}", self.ptr, self.count, std::mem::size_of::<T>());
        unsafe { dealloc(self.ptr as *mut u8, layout); }
    }
}


#[derive(Clone)]
pub struct MemShared<T> {
    mem: Rc<RefCell<MemPtr<T>>>,
    offset: isize,
}

impl<T> MemShared<T> {
    /// Make a new instance which pointer is offset by a length `offset * std::mem::size_of::<T>()`.
    pub fn offset(&self, offset: i32) -> Self {
        MemShared {
            mem: self.mem.clone(),
            offset: self.offset + offset as isize,
        }
    }
}


/// Manages memory allocation <strike>and synchronization between the host (CPU) and device (GPU)</strike>
pub struct SyncedMemory<T: Sized> {
    cpu_mem: Option<Rc<RefCell<MemPtr<T>>>>,
    count: usize,
    cpu_offset: isize,
}

impl<T: Sized> Default for SyncedMemory<T> {
    fn default() -> Self {
        SyncedMemory {
            cpu_mem: Default::default(),
            count: 0,
            cpu_offset: 0,
        }
    }
}

impl<T: Sized> SyncedMemory<T> {
    pub fn new_uninit() -> Self {
        Default::default()
    }

    /// Construct an instance without allocating memory. Note that `count` is the num of item of type `T`,
    /// so the actual memory size in bytes is `std::mem::size_of::<T>() * count`.
    pub fn new(count: usize) -> Self {
        SyncedMemory {
            cpu_mem: Default::default(),
            count,
            cpu_offset: 0
        }
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn bytes_size(&self) -> usize {
        std::mem::size_of::<T>() * self.count
    }

    fn sync_to_cpu(&mut self) {
        if let Option::None = self.cpu_mem {
            trace!("Synced CPU memory type from uninitialized to alloc.");
            self.cpu_mem = Some(Rc::new(RefCell::new(MemPtr::new(self.count))));
        }
    }

    pub fn cpu_data(&mut self) -> &[T] {
        self.sync_to_cpu();
        let (ptr, _) = RefCell::borrow(self.cpu_mem.as_ref().unwrap()).raw_parts();
        unsafe { std::slice::from_raw_parts(ptr.offset(self.cpu_offset), self.count) }
    }

    pub fn cpu_data_raw(&mut self) -> (*const T, usize) {
        self.sync_to_cpu();
        let (ptr, _) = self.cpu_mem.as_ref().unwrap().as_ref().borrow().raw_parts();
        (unsafe { ptr.offset(self.cpu_offset) }, self.count)
    }

    pub fn cpu_data_shared(&mut self) -> MemShared<T> {
        self.sync_to_cpu();
        MemShared {
            mem: Rc::clone(self.cpu_mem.as_ref().unwrap()),
            offset: self.cpu_offset,
        }
    }

    pub fn try_map_cpu_data<F, U>(&self, f: F) -> Option<U> where F: FnOnce(&[T]) -> U {
        self.cpu_mem.as_ref().map(|ptr| {
            let (ptr, _) = RefCell::borrow((*ptr).as_ref()).raw_parts();
            f(unsafe { std::slice::from_raw_parts(ptr.offset(self.cpu_offset), self.count) })
        })
    }

    pub fn mutable_cpu_data(&mut self) -> &mut [T] {
        self.sync_to_cpu();
        let (ptr, _) = RefCell::borrow_mut(self.cpu_mem.as_ref().unwrap()).raw_parts_mut();
        unsafe { std::slice::from_raw_parts_mut(ptr.offset(self.cpu_offset), self.count) }
    }

    pub fn mutable_cpu_data_raw(&mut self) -> (*mut T, usize) {
        self.sync_to_cpu();
        let (ptr, _) = self.cpu_mem.as_ref().unwrap().borrow_mut().raw_parts_mut();
        (unsafe { ptr.offset(self.cpu_offset) }, self.count)
    }

    pub fn try_map_cpu_mut_data<F, U>(&mut self, f: F) -> Option<U> where F: FnOnce(&mut [T]) -> U {
        self.cpu_mem.as_ref().map(|ptr| {
            let (ptr, _) = RefCell::borrow_mut((*ptr).as_ref()).raw_parts_mut();
            f(unsafe { std::slice::from_raw_parts_mut(ptr.offset(self.cpu_offset), self.count) })
        })
    }

    pub fn set_cpu_data(&mut self, data: &MemShared<T>) {
        let mem_ptr = data.mem.as_ptr();
        let &MemPtr { ptr, count } = unsafe { &*mem_ptr };

        trace!("Set a borrowed slice of CPU memory. ptr: {:?}, len: {} * {}; offset: {}",
               ptr, count, std::mem::size_of::<T>(), data.offset);
        if data.offset < 0 {
            panic!("Set a borrowed memory but which offset({}) < 0.", data.offset);
        }
        if self.count + data.offset as usize > count {
            panic!("Set a slice which length ({} - offset({}) = {}) less than the memory need ({}).",
                   count, data.offset, count as isize - data.offset, self.count);
        }

        self.cpu_mem = Some(Rc::clone(&data.mem));
        self.cpu_offset = data.offset;
    }
}


pub struct ArcSyncedMemory<T: Sized> {
    cpu_mem: Option<Arc<Mutex<MemPtr<T>>>>,
    count: usize,
    cpu_offset: isize,
}

impl<T: Sized> ArcSyncedMemory<T> {
    pub fn new() -> Self {
        Self {
            cpu_mem: Default::default(),
            count: 0,
            cpu_offset: 0,
        }
    }

    pub fn from(mut mem: SyncedMemory<T>) -> Result<Self, SyncedMemory<T>> {
        let cpu_mem = mem.cpu_mem.map(|r| Rc::try_unwrap(r));
        if cpu_mem.as_ref().map_or(false, |r| r.is_err()) {
            mem.cpu_mem = cpu_mem.map(|r| r.err().unwrap());
            return Result::Err(mem);
        }

        let cpu_mem = cpu_mem.map(
            |r| Arc::new(Mutex::new(r.ok().unwrap().into_inner()))
        );
        let arc_mem = ArcSyncedMemory {
            cpu_mem,
            count: mem.count,
            cpu_offset: mem.cpu_offset,
        };
        Result::Ok(arc_mem)
    }

    pub fn into_mem(mut self) -> Result<SyncedMemory<T>, Self> {
        let cpu_mem = self.cpu_mem.map(|a| Arc::try_unwrap(a));
        if cpu_mem.as_ref().map_or(false, |r| r.is_err()) {
            self.cpu_mem = cpu_mem.map(|r| r.err().unwrap());
            return Result::Err(self);
        }

        let cpu_mem = cpu_mem.map(
            |r| Rc::new(RefCell::new(r.ok().unwrap().into_inner().unwrap()))
        );
        let mem = SyncedMemory {
            cpu_mem,
            count: self.count,
            cpu_offset: self.cpu_offset,
        };
        Result::Ok(mem)
    }
}


#[cfg(test)]
use test_env_log::test;

#[test]
fn mem_ptr_test_new() {
    let _ = MemPtr::<u8>::new(54);
}

#[test]
fn synced_mem_test_uninit() {
    let mut s = SyncedMemory::new_uninit();
    let slice: &[i32] = s.cpu_data();
    info!("New uninitialized memory, ptr: {:?}, len: {}", slice.as_ptr(), slice.len());
}

#[test]
fn synced_mem_test_new() {
    let mut s = SyncedMemory::new(78);
    {
        let mut slice = s.mutable_cpu_data();
        info!("Get mutable slice from SyncedMemory: {:#?}", slice);
        let mut count = 0u8;
        for x in slice {
            count += 1;
            *x = count;
        }
    }

    let slice = s.cpu_data();
    info!("Get const slice from SyncedMemory: {:#?}", slice);
}

#[test]
fn synced_mem_test_slice() {
    let mem = MemShared {
        mem: Rc::new(RefCell::new(MemPtr::new(12))),
        offset: 2,
    };
    {
        let mut s = SyncedMemory::new(9);
        info!("Set slice data");
        s.set_cpu_data(&mem);

        let mut slice = s.mutable_cpu_data();
        info!("Get mutable slice from SyncedMemory: {:#?}", slice);
        let mut count = 2u8;
        for x in slice {
            *x = count * 2u8;
            count += 1;
        }
    }

    info!("Print original slice: {:#?}", RefCell::borrow(&mem.mem).as_slice());
}
