use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use crate::common::{CaffeBrew, Caffe};
use crate::util::rng::caffe_rng_rand;


#[derive(Default)]
pub struct InternalThreadImpl {
    pub thread: Option<JoinHandle<()>>,
    pub interrupt: Arc<AtomicBool>,
}

pub struct CancelToken {
    interrupt: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new(ir: &Arc<AtomicBool>) -> Self {
        CancelToken {
            interrupt: Arc::clone(ir)
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.interrupt.load(Ordering::Relaxed)
    }
}


/// Trait encapsulate std::thread for use in base class. The child class will acquire the
/// ability to run a single thread, by implementing the virtual function `internal_thread_entry`.
pub trait InternalThread {
    type EntryData: Send + 'static;

    fn get_thread(&self) -> &InternalThreadImpl;

    fn get_thread_mut(&mut self) -> &mut InternalThreadImpl;

    fn get_entry_data(&mut self) -> Self::EntryData;

    /// Implement this method in your subclass with the code you want your thread to run.
    fn internal_thread_entry(token: CancelToken, data: Self::EntryData) where Self: Sized;
}

pub trait InternalThreadLaunch: InternalThread {
    /// Caffe's thread local state will be initialized using the current
    /// thread values, e.g. device id, solver index etc. The random seed
    /// is initialized using caffe_rng_rand.
    fn start_internal_thread(&mut self);
}

impl<T: Sized + InternalThread> InternalThreadLaunch for T {
    fn start_internal_thread(&mut self) {
        assert!(self.get_thread().thread.is_none(), "Threads should persist and not be restarted.");

        let _device = 0;
        let mode = Caffe::mode();
        let rand_seed = caffe_rng_rand();
        let solver_count = Caffe::solver_count();
        let solver_rank = Caffe::solver_rank();
        let multiprocess = Caffe::multiprocess();
        let data = self.get_entry_data();

        let th = self.get_thread_mut();
        let token = CancelToken::new(&th.interrupt);
        th.thread = Some(std::thread::spawn(move || {
            Caffe::set_mode(mode);
            Caffe::set_random_seed(rand_seed as u64);
            Caffe::set_solver_count(solver_count);
            Caffe::set_solver_rank(solver_rank);
            Caffe::set_multiprocess(multiprocess);
            Self::internal_thread_entry(token, data);
        }));
    }
}

impl<E: Send + 'static> dyn InternalThread<EntryData = E> {
    /// Will block until the internal thread has exited.
    pub fn stop_internal_thread(&mut self) {
        let th = self.get_thread_mut();
        th.interrupt.store(true, Ordering::Relaxed);
        let handle = th.thread.take();
        handle.map(|t| t.join().unwrap());
    }

    pub fn is_started(&self) -> bool {
        self.get_thread().thread.is_some()
    }
}
