use std::alloc::{alloc_zeroed, dealloc, Layout};

enum MemoryRef<'a, T: Sized> {
    Uninitialized,
    Owned { ptr: *mut T, layout: Layout },
    Borrowed { slice: &'a mut [T] },
}

impl<T: Sized> Drop for MemoryRef<'_, T> {
    fn drop(&mut self) {
        if let &mut MemoryRef::Owned { ptr, layout } = self {
            trace!("dealloc owned memory, ptr: {:?}, layout: {:?}", ptr, layout);
            unsafe { dealloc(ptr as *mut u8, layout); }
        }
    }
}

impl<'a, T: Sized> MemoryRef<'a, T> {
    pub fn new(count: usize) -> Self {
        let layout = Layout::array::<T>(count).unwrap();
        trace!("alloc owned memory, layout: {:?}", layout);
        MemoryRef::Owned { ptr: unsafe { alloc_zeroed(layout) as *mut T }, layout }
    }

    pub fn new_uninit() -> Self {
        MemoryRef::Uninitialized
    }

    pub fn borrow(slice: &'a mut [T]) -> Self {
        MemoryRef::Borrowed { slice }
    }
}

impl<T: Sized> Default for MemoryRef<'_, T> {
    fn default() -> Self {
        MemoryRef::Uninitialized
    }
}


/// Manages memory allocation <strike>and synchronization between the host (CPU) and device (GPU)</strike>
pub struct SyncedMemory<'a, T: Sized> {
    cpu_mem: MemoryRef<'a, T>,
    count: usize,
}

impl<T: Sized> Default for SyncedMemory<'_, T> {
    fn default() -> Self {
        SyncedMemory {
            cpu_mem: Default::default(),
            count: 0
        }
    }
}

impl<T: Sized> SyncedMemory<'_, T> {
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
        if let MemoryRef::Uninitialized = self.cpu_mem {
            trace!("Synced CPU memory type from uninitialized to alloc.");
            self.cpu_mem = MemoryRef::new(self.count);
        }
    }

    pub fn cpu_data<'a>(&'a mut self) -> &'a [T] {
        // note: let caller call this method manually.
        self.sync_to_cpu();
        match self.cpu_mem {
            MemoryRef::Owned { ptr, .. } => unsafe { std::slice::from_raw_parts::<'a, T>(ptr, self.count) },
            MemoryRef::Borrowed { ref slice } => *slice,
            MemoryRef::Uninitialized => panic!("Unreachable code!")
        }
    }

    pub fn mutable_cpu_data<'a>(&'a mut self) -> &'a mut [T] {
        self.sync_to_cpu();
        match self.cpu_mem {
            MemoryRef::Owned { ptr, .. } => unsafe { std::slice::from_raw_parts_mut::<'a, T>(ptr, self.count) },
            MemoryRef::Borrowed { ref mut slice } => *slice,
            MemoryRef::Uninitialized => panic!("Unreachable code!")
        }
    }
}

impl<'a, T: Sized> SyncedMemory<'a, T> {
    pub fn set_cpu_data(&mut self, data: &'a mut [T]) {
        trace!("Set a borrowed slice of CPU memory. ptr: {:?}, len: {} * {}", data.as_ptr(), data.len(), std::mem::size_of::<T>());
        if self.count > data.len() {
            panic!("Set a slice which length ({}) less than the memory need ({}).", data.len(), self.count);
        }
        self.cpu_mem = MemoryRef::borrow(data);
    }
}


#[cfg(test)]
use test_env_log::test;

#[test]
fn memory_ref_test_new() {
    let _ = MemoryRef::<u8>::new(54);
}

#[test]
fn synced_mem_test_uninit() {
    let mut s = SyncedMemory::new_uninit();
    s.sync_to_cpu();
    let slice: &[i32] = s.cpu_data();
    info!("New uninitialized memory, ptr: {:?}, len: {}", slice.as_ptr(), slice.len());
}

#[test]
fn synced_mem_test_new() {
    let mut s = SyncedMemory::new(78);
    let slice = s.mutable_cpu_data();
    info!("Get mutable slice from SyncedMemory: {:#?}", slice);
    let mut count = 0u8;
    for x in slice {
        count += 1;
        *x = count;
    }

    let slice = s.cpu_data();
    info!("Get const slice from SyncedMemory: {:#?}", slice);
}

#[test]
fn synced_mem_test_slice() {
    let mut vec = Vec::with_capacity(12);
    {
        let mut s = SyncedMemory::new(9);
        vec.resize(12, 0u8);
        info!("Set slice data");
        s.set_cpu_data(&mut vec);

        let slice = s.mutable_cpu_data();
        info!("Get mutable slice from SyncedMemory: {:#?}", slice);
        let mut count = 2u8;
        for x in slice {
            *x = count * 2u8;
            count += 1;
        }
    }

    info!("Print original slice: {:#?}", &vec);
}
