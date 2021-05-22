use std::cell::RefCell;
use std::rc::Rc;

use mt19937::MT19937;
use rand::{RngCore, thread_rng, SeedableRng};


#[derive(Copy, Clone)]
pub enum CaffeBrew {
    CPU,
    GPU,
}

pub struct CaffeRng {
    rng: MT19937,
}

// Random seeding. The c++ source only read '/dev/urandom' to fetch a seed.
// Use `ThreadRng` to get a cross-platform Cryptographically Secure-PRNG
fn cluster_seed_gen() -> u64 {
    thread_rng().next_u64()
}

impl CaffeRng {
    pub fn new() -> Self {
        CaffeRng {
            rng: MT19937::seed_from_u64(cluster_seed_gen()),
        }
    }

    pub fn new_with_seed(seed: u64) -> Self {
        CaffeRng {
            rng: MT19937::seed_from_u64(seed),
        }
    }

    pub fn generator(&mut self) -> &mut dyn RngCore {
        &mut self.rng
    }
}

pub struct Caffe {
    mode: CaffeBrew,
    solver_count: i32,
    solver_rank: i32,
    multiprocess: bool,
    random_generator: Option<Rc<RefCell<CaffeRng>>>,
}

impl Caffe {
    const fn new() -> Self {
        Caffe {
            mode: CaffeBrew::CPU,
            solver_count: 1,
            solver_rank: 0,
            multiprocess: false,
            random_generator: None,
        }
    }
}

thread_local! {
    static CAFFE: RefCell<Caffe> = RefCell::new(Caffe::new());
}

impl Caffe {
    pub fn mode() -> CaffeBrew {
        CAFFE.with(|f| {
            f.borrow().mode
        })
    }

    pub fn set_mode(mode: CaffeBrew) {
        CAFFE.with(|f| {
            (*f.borrow_mut()).mode = mode;
        });
    }

    pub fn set_random_seed(seed: u64) {
        // RNG seed
        CAFFE.with(|f| {
            f.borrow_mut().random_generator.replace(Rc::new(RefCell::new(CaffeRng::new_with_seed(seed))));
        });
    }

    pub fn rng() -> Rc<RefCell<CaffeRng>> {
        CAFFE.with(|f| {
            f.borrow_mut().random_generator
                .get_or_insert_with(|| Rc::new(RefCell::new(CaffeRng::new())))
                .clone()
        })
    }

    pub fn rng_rand() -> u32 {
        CAFFE.with(|f| {
            f.borrow_mut().random_generator
                .get_or_insert_with(|| Rc::new(RefCell::new(CaffeRng::new())))
                .as_ref().borrow_mut()
                .rng.next_u32()
        })
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
            f.borrow().solver_count
        })
    }

    pub fn set_solver_count(val: i32) {
        CAFFE.with(|f| {
            f.borrow_mut().solver_count = val;
        });
    }

    pub fn solver_rank() -> i32 {
        CAFFE.with(|f| {
            f.borrow().solver_rank
        })
    }

    pub fn set_solver_rank(val: i32) {
        CAFFE.with(|f| {
            f.borrow_mut().solver_rank = val;
        });
    }

    pub fn multiprocess() -> bool {
        CAFFE.with(|f| {
            f.borrow().multiprocess
        })
    }

    pub fn set_multiprocess(val: bool) {
        CAFFE.with(|f| {
            f.borrow_mut().multiprocess = val;
        });
    }

    pub fn root_solver() -> bool {
        CAFFE.with(|f| {
            f.borrow().solver_rank
        }) == 0
    }
}
