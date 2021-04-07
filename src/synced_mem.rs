use std::alloc::{alloc_zeroed, dealloc, Layout};

enum MemoryRef<'a> {
    Uninitialized,
    Owned { ptr: *mut u8, layout: Layout },
    Borrowed { slice: &'a mut [u8] },
}

impl Drop for MemoryRef<'_> {
    fn drop(&mut self) {
        if let &mut MemoryRef::Owned { ptr, layout } = self {
            trace!("dealloc owned memory, ptr: {:?}, layout: {:?}", ptr, layout);
            unsafe { dealloc(ptr, layout); }
        }
    }
}

impl<'a> MemoryRef<'a> {
    pub fn new(size: usize) -> Self {
        let layout = Layout::array::<u8>(size).unwrap();
        trace!("alloc owned memory, layout: {:?}", layout);
        MemoryRef::Owned { ptr: unsafe { alloc_zeroed(layout) }, layout }
    }

    pub fn new_uninit() -> Self {
        MemoryRef::Uninitialized
    }

    pub fn borrow(slice: &'a mut [u8]) -> Self {
        MemoryRef::Borrowed { slice }
    }
}

impl Default for MemoryRef<'_> {
    fn default() -> Self {
        MemoryRef::Uninitialized
    }
}


/// Manages memory allocation <strike>and synchronization between the host (CPU) and device (GPU)</strike>
#[derive(Default)]
pub struct SyncedMemory<'a> {
    cpu_mem: MemoryRef<'a>,
    size: usize,
}

impl SyncedMemory<'_> {
    pub fn new_uninit() -> Self {
        Default::default()
    }

    pub fn new(size: usize) -> Self {
        SyncedMemory {
            cpu_mem: Default::default(),
            size,
        }
    }

    fn sync_to_cpu(&mut self) {
        if let MemoryRef::Uninitialized = self.cpu_mem {
            trace!("Synced CPU memory type from uninitialized to alloc.");
            self.cpu_mem = MemoryRef::new(self.size);
        }
    }

    pub fn cpu_data<'a>(&'a mut self) -> &'a [u8] {
        self.sync_to_cpu();
        match self.cpu_mem {
            MemoryRef::Owned { ptr, .. } => unsafe { std::slice::from_raw_parts::<'a, u8>(ptr, self.size) },
            MemoryRef::Borrowed { ref mut slice } => *slice,
            MemoryRef::Uninitialized => panic!("Unreachable code!")
        }
    }

    pub fn mutable_cpu_data<'a>(&'a mut self) -> &'a mut [u8] {
        self.sync_to_cpu();
        match self.cpu_mem {
            MemoryRef::Owned { ptr, .. } => unsafe { std::slice::from_raw_parts_mut::<'a, u8>(ptr, self.size) },
            MemoryRef::Borrowed { ref mut slice } => *slice,
            MemoryRef::Uninitialized => panic!("Unreachable code!")
        }
    }
}

impl<'a> SyncedMemory<'a> {
    pub fn set_cpu_data(&mut self, data: &'a mut [u8]) {
        trace!("Set a borrowed slice of CPU memory. ptr: {:?}, len: {}", data.as_ptr(), data.len());
        if self.size > data.len() {
            panic!("Set a slice which length ({}) less than the memory need ({}).", data.len(), self.size);
        }
        self.cpu_mem = MemoryRef::borrow(data);
    }
}


#[cfg(test)]
use test_env_log::test;

#[test]
fn memory_ref_test_new() {
    let _ = MemoryRef::new(54);
}

#[test]
fn synced_mem_test_uninit() {
    let mut s = SyncedMemory::new_uninit();
    let slice = s.cpu_data();
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
