//! Lock-free memory pool for recycling byte buffers across concurrent encoding threads.

use std::cell::RefCell;
use std::sync::Arc;
use bytes::{Bytes, BytesMut};
use crossbeam_queue::ArrayQueue;

/// Capacity of the global lock-free pool.
const POOL_CAPACITY: usize = 32;
/// Initial buffer allocation size (1 MB).
const DEFAULT_BUFFER_SIZE: usize = 1024 * 1024;

thread_local! {
    static LOCAL_SCRATCH: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(DEFAULT_BUFFER_SIZE));
}

/// Global shared pool of reusable `BytesMut` buffers.
#[derive(Debug, Clone)]
pub struct BufferPool {
    queue: Arc<ArrayQueue<BytesMut>>,
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferPool {
    /// Creates a new global buffer pool.
    pub fn new() -> Self {
        Self {
            queue: Arc::new(ArrayQueue::new(POOL_CAPACITY)),
        }
    }

    /// Acquires a clean `BytesMut` buffer with at least `min_capacity` bytes.
    pub fn acquire(&self, min_capacity: usize) -> BytesMut {
        if let Some(mut buf) = self.queue.pop() {
            buf.clear();
            if buf.capacity() < min_capacity {
                buf.reserve(min_capacity - buf.capacity());
            }
            buf
        } else {
            BytesMut::with_capacity(min_capacity.max(DEFAULT_BUFFER_SIZE))
        }
    }

    /// Returns a used buffer back to the pool for reuse.
    pub fn release(&self, mut buf: BytesMut) {
        buf.clear();
        let _ = self.queue.push(buf);
    }
}

/// Helper function to execute a closure with thread-local scratch memory.
#[inline(always)]
pub fn with_thread_scratch<F, R>(min_size: usize, f: F) -> R
where
    F: FnOnce(&mut Vec<u8>) -> R,
{
    LOCAL_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.capacity() < min_size {
            scratch.reserve(min_size - scratch.capacity());
        }
        scratch.clear();
        f(&mut scratch)
    })
}
