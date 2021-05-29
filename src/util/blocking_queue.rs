//! Simple blocking queue implementation which take [this repo](https://github.com/Julian6bG/rust-blockinqueue)
//! as a reference.

use std::sync::{Arc, Mutex};
use std::sync::mpsc::{Sender, Receiver, channel};


#[derive(Clone)]
pub struct BlockingQueue<T> {
    sender: Sender<T>,
    receiver: Arc<Mutex<Receiver<T>>>,
}

impl<T> BlockingQueue<T> {
    pub fn new() -> Self {
        let (sender, receiver) = channel();
        Self {
            sender,
            receiver: Arc::new(Mutex::new(receiver)),
        }
    }

    pub fn push(&self, v: T) {
        self.sender.send(v).unwrap();
    }

    pub fn pop(&self) -> T {
        self.receiver.lock().unwrap().recv().unwrap()
    }

    pub fn try_pop(&self) -> Option<T> {
        self.receiver.lock().unwrap().try_recv().ok()
    }
}
