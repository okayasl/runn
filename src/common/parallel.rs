use crossbeam_channel::{unbounded, Receiver, Sender};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

pub struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    job_sender: Option<Sender<Job>>,
    remaining_jobs: Arc<(Mutex<usize>, Condvar)>,
}

impl ThreadPool {
    pub fn new(num_threads: usize) -> Self {
        let (job_sender, job_receiver) = unbounded::<Job>();
        let job_receiver = Arc::new(job_receiver);
        let remaining_jobs = Arc::new((Mutex::new(0), Condvar::new()));

        let mut workers = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let receiver = Arc::clone(&job_receiver);
            let remaining = Arc::clone(&remaining_jobs);
            let handle = thread::spawn(move || {
                while let Ok(job) = receiver.recv() {
                    let _ = catch_unwind(AssertUnwindSafe(|| {
                        job();
                    }));
                    let (lock, cvar) = &*remaining;
                    let mut rem = lock.lock().unwrap();
                    *rem -= 1;
                    cvar.notify_all();
                }
            });
            workers.push(handle);
        }

        Self {
            workers,
            job_sender: Some(job_sender),
            remaining_jobs,
        }
    }

    pub fn submit<R, F>(&self, job: F) -> Receiver<R>
    where
        R: Send + 'static,
        F: FnOnce() -> R + Send + 'static,
    {
        let (result_sender, result_receiver) = unbounded();
        let job_wrapper = Box::new(move || {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                let result = job();
                let _ = result_sender.send(result);
            }));
        });

        let (lock, _) = &*self.remaining_jobs;
        *lock.lock().unwrap() += 1;
        self.job_sender
            .as_ref()
            .unwrap()
            .send(job_wrapper)
            .expect("Failed to submit job");

        result_receiver
    }

    pub fn join(&self) {
        let (lock, cvar) = &*self.remaining_jobs;
        let mut remaining = lock.lock().unwrap();
        while *remaining > 0 {
            remaining = cvar.wait(remaining).unwrap();
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        drop(self.job_sender.take());
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    #[test]
    fn test_basic_execution() {
        let pool = ThreadPool::new(4);
        let receiver = pool.submit(|| 42);
        pool.join();
        let result = receiver.recv().unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_ordered_results() {
        let pool = ThreadPool::new(4);
        let mut receivers = Vec::new();

        for i in 0..10 {
            receivers.push(pool.submit(move || i * 2));
        }

        pool.join();
        let results: Vec<_> = receivers.into_iter().map(|r| r.recv().unwrap()).collect();

        for (i, val) in results.iter().enumerate() {
            assert_eq!(*val, (i as i32) * 2);
        }
    }

    #[test]
    fn test_parallel_execution_speedup() {
        let pool = ThreadPool::new(4);
        let now = Instant::now();
        let mut receivers = Vec::new();

        for _ in 0..4 {
            receivers.push(pool.submit(|| {
                std::thread::sleep(Duration::from_millis(200));
                1
            }));
        }

        pool.join();
        let elapsed = now.elapsed();

        assert!(elapsed < Duration::from_millis(800), "Tasks were not parallelized");
        let sum: i32 = receivers.into_iter().map(|r| r.recv().unwrap()).sum();
        assert_eq!(sum, 4);
    }

    #[test]
    fn test_mixed_types() {
        let pool = ThreadPool::new(4);

        let r1 = pool.submit(|| 100);
        let r2 = pool.submit(|| String::from("hello").to_uppercase());
        let r3 = pool.submit(|| vec![1, 2, 3].into_iter().sum::<i32>());

        pool.join();

        assert_eq!(r1.recv().unwrap(), 100);
        assert_eq!(r2.recv().unwrap(), "HELLO");
        assert_eq!(r3.recv().unwrap(), 6);
    }

    #[test]
    fn test_thread_safety_concurrent_submits() {
        let pool = Arc::new(ThreadPool::new(4));
        let results = Arc::new(Mutex::new(vec![]));
        let mut handles = vec![];

        for i in 0..10 {
            let pool = Arc::clone(&pool);
            let results = Arc::clone(&results);
            let handle = thread::spawn(move || {
                let recv = pool.submit(move || i * i);
                let res = recv.recv().unwrap();
                results.lock().unwrap().push(res);
            });
            handles.push(handle);
        }

        for h in handles {
            h.join().unwrap();
        }

        pool.join();

        let results = results.lock().unwrap();
        assert_eq!(results.len(), 10);
        for r in results.iter() {
            assert!(results.contains(r));
        }
    }
}

// pub struct ThreadPool<T, R> {
//     workers: Vec<thread::JoinHandle<()>>,
//     sender: Option<Sender<Option<T>>>,
//     results: Receiver<R>,
// }

// impl<T: Send + 'static, R: Send + 'static> ThreadPool<T, R> {
//     /// Creates a new thread pool with the specified number of threads.
//     pub fn new<F>(num_threads: usize, job_handler: F) -> Self
//     where
//         F: Fn(T) -> R + Send + Sync + 'static,
//     {
//         let (job_sender, job_receiver) = unbounded::<Option<T>>();
//         let (result_sender, result_receiver) = unbounded::<R>();

//         let mut workers = Vec::with_capacity(num_threads);
//         let job_handler = Arc::new(job_handler);

//         for _ in 0..num_threads {
//             let job_receiver = job_receiver.clone();
//             let result_sender = result_sender.clone();
//             let job_handler = Arc::clone(&job_handler);

//             let worker = thread::spawn(move || {
//                 while let Ok(job) = job_receiver.recv() {
//                     if let Some(job) = job {
//                         let result = job_handler(job);
//                         if result_sender.send(result).is_err() {
//                             break; // Stop if result channel is closed
//                         }
//                     } else {
//                         break; // Shutdown signal
//                     }
//                 }
//             });

//             workers.push(worker);
//         }

//         Self {
//             workers,
//             sender: Some(job_sender),
//             results: result_receiver,
//         }
//     }

//     /// Sends a job to the thread pool.
//     pub fn send(&self, job: T) {
//         if let Some(sender) = &self.sender {
//             sender.send(Some(job)).expect("Failed to send job");
//         }
//     }

//     /// Collects results from the thread pool.
//     pub fn collect_results(&self, num_jobs: usize) -> Vec<R> {
//         let mut results = Vec::with_capacity(num_jobs);
//         for _ in 0..num_jobs {
//             match self.results.recv() {
//                 Ok(result) => results.push(result),
//                 Err(_) => break, // Stop if channel is disconnected
//             }
//         }
//         results
//     }
// }

// impl<T, R> Drop for ThreadPool<T, R> {
//     fn drop(&mut self) {
//         if let Some(sender) = self.sender.take() {
//             for _ in &self.workers {
//                 sender.send(None).expect("Failed to send shutdown signal");
//             }
//         }

//         for worker in self.workers.drain(..) {
//             worker.join().expect("Worker thread panicked");
//         }
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_thread_pool() {
//         let pool = ThreadPool::new(4, |x: i32| x * 2);
//         let num_jobs: usize = 10;

//         for i in 0..num_jobs {
//             pool.send(i as i32);
//         }

//         let results = pool.collect_results(num_jobs);
//         assert_eq!(results.len(), num_jobs);
//         for (i, result) in results.iter().enumerate() {
//             assert_eq!(*result, (i as i32) * 2);
//         }
//     }

//     #[test]
//     fn test_thread_pool_with_strings() {
//         let pool = ThreadPool::new(4, |x: String| x.len());
//         let num_jobs = 10;

//         for i in 0..num_jobs {
//             pool.send(format!("test{}", i));
//         }

//         let results = pool.collect_results(num_jobs);
//         assert_eq!(results.len(), num_jobs);
//         for (i, result) in results.iter().enumerate() {
//             assert_eq!(*result, format!("test{}", i).len());
//         }
//     }
// }
