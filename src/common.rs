use std::cell::RefCell;

use static_init::dynamic;


pub enum CaffeBrew {
    CPU,
    GPU,
}

pub struct Caffe {
    // shared_ptr rng
    mode: CaffeBrew,
    solver_count: i32,
    solver_rank: i32,
    multiprocess: bool,
}

impl Caffe {
    const fn new() -> Self {
        Caffe {
            mode: CaffeBrew::CPU,
            solver_count: 1,
            solver_rank: 0,
            multiprocess: false
        }
    }
}

thread_local! {
    static CAFFE: RefCell<Caffe> = RefCell::new(Caffe::new());
}

impl Caffe {
    pub fn mode() -> CaffeBrew {
        CAFFE.with(|f| {
            *f.borrow().mode
        })
    }

    pub fn set_mode(mode: CaffeBrew) {
        CAFFE.with(|f| {
            *f.borrow_mut().mode = mode;
        });
    }

    pub fn set_random_seed(seed: u32) {
        // todo: rng
    }

    pub fn set_device(device_id: i32) {
        // todo: gpu
        unimplemented!();
    }

    pub fn device_query() {
        // todo: gpu
        unimplemented!();
    }

    pub fn check_device(device_id: i32) -> bool {
        // todo: gpu
        unimplemented!();
    }

    pub fn find_device(start_id: i32) -> i32 {
        // todo: gpu
        unimplemented!();
    }

    pub fn solver_count() -> i32 {
        CAFFE.with(|f| {
            *f.borrow().solver_count
        })
    }

    pub fn set_solver_count(val: i32) {
        CAFFE.with(|f| {
            *f.borrow_mut().solver_count = val;
        });
    }

    pub fn solver_rank() -> i32 {
        CAFFE.with(|f| {
            *f.borrow().solver_rank
        })
    }

    pub fn set_solver_rank(val: i32) {
        CAFFE.with(|f| {
            *f.borrow_mut().solver_rank = val;
        });
    }

    pub fn multiprocess() -> bool {
        CAFFE.with(|f| {
            *f.borrow().multiprocess
        })
    }

    pub fn set_multiprocess(val: bool) {
        CAFFE.with(|f| {
            *f.borrow_mut().multiprocess = val;
        });
    }

    pub fn root_solver() -> bool {
        CAFFE.with(|f| {
            *f.borrow().solver_rank
        }) == 0
    }
}
