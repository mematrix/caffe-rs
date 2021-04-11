use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::rc::Rc;
use std::cell::{RefCell, Ref, RefMut};


pub struct MemPtr<T> {
    ptr: *mut T,
    count: usize,
}

impl<T> MemPtr<T> {
    pub fn new(count: usize) -> Self {
        let layout = Layout::array::<T>(count).unwrap();
        trace!("alloc owned memory, layout: {:?}", layout);
        MemPtr {
            ptr: unsafe { alloc_zeroed(layout) as *mut T },
            count,
        }
    }

    pub fn raw_parts(&self) -> (*mut T, usize) {
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
    mem: Rc<RefCell<MemPtr<T>>>
}


/// Manages memory allocation <strike>and synchronization between the host (CPU) and device (GPU)</strike>
pub struct SyncedMemory<T: Sized> {
    cpu_mem: Option<Rc<RefCell<MemPtr<T>>>>,
    count: usize,
}

impl<T: Sized> Default for SyncedMemory<T> {
    fn default() -> Self {
        SyncedMemory {
            cpu_mem: Default::default(),
            count: 0,
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

    pub fn cpu_data(&mut self) -> Ref<MemPtr<T>> {
        // note: let caller call this method manually.
        self.sync_to_cpu();
        RefCell::borrow(self.cpu_mem.as_ref().unwrap())
    }

    pub fn cpu_data_shared(&mut self) -> MemShared<T> {
        self.sync_to_cpu();
        MemShared {
            mem: Rc::clone(self.cpu_mem.as_ref().unwrap())
        }
    }

    pub fn try_map_cpu_data<F, U>(&self, f: F) -> Option<U> where F: FnOnce(&[T]) -> U {
        self.cpu_mem.as_ref().map(|ptr| {
            let data = RefCell::borrow((*ptr).as_ref());
            f(data.as_slice())
        })
    }

    pub fn mutable_cpu_data(&mut self) -> RefMut<MemPtr<T>> {
        self.sync_to_cpu();
        RefCell::borrow_mut(self.cpu_mem.as_ref().unwrap())
    }

    pub fn set_cpu_data(&mut self, data: &MemShared<T>) {
        let mem_ptr = data.mem.as_ptr();
        let &MemPtr { ptr, count } = unsafe { &*mem_ptr };

        trace!("Set a borrowed slice of CPU memory. ptr: {:?}, len: {} * {}", ptr, count, std::mem::size_of::<T>());
        if self.count > count {
            panic!("Set a slice which length ({}) less than the memory need ({}).", count, self.count);
        }

        self.cpu_mem = Some(Rc::clone(&data.mem));
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
    let slice = s.cpu_data();
    let slice: &[i32] = slice.as_slice();
    info!("New uninitialized memory, ptr: {:?}, len: {}", slice.as_ptr(), slice.len());
}

#[test]
fn synced_mem_test_new() {
    let mut s = SyncedMemory::new(78);
    {
        let mut slice = s.mutable_cpu_data();
        let slice = slice.as_mut_slice();
        info!("Get mutable slice from SyncedMemory: {:#?}", slice);
        let mut count = 0u8;
        for x in slice {
            count += 1;
            *x = count;
        }
    }

    let slice = s.cpu_data();
    let slice = slice.as_slice();
    info!("Get const slice from SyncedMemory: {:#?}", slice);
}

#[test]
fn synced_mem_test_slice() {
    let mem = MemShared {
        mem: Rc::new(RefCell::new(MemPtr::new(12)))
    };
    {
        let mut s = SyncedMemory::new(9);
        info!("Set slice data");
        s.set_cpu_data(&mem);

        let mut slice = s.mutable_cpu_data();
        let slice = slice.as_mut_slice();
        info!("Get mutable slice from SyncedMemory: {:#?}", slice);
        let mut count = 2u8;
        for x in slice {
            *x = count * 2u8;
            count += 1;
        }
    }

    info!("Print original slice: {:#?}", RefCell::borrow(&mem.mem).as_slice());
}
