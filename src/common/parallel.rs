use std::sync::{mpsc, Arc, Mutex};
use std::thread;

pub struct ThreadPool<T, R> {
    workers: Vec<thread::JoinHandle<()>>,
    sender: Option<mpsc::Sender<Option<T>>>, // Use Option to signal shutdown
    results: Arc<Mutex<mpsc::Receiver<R>>>,
}

impl<T: Send + 'static, R: Send + 'static> ThreadPool<T, R> {
    /// Creates a new thread pool with the specified number of threads.
    pub fn new<F>(num_threads: usize, job_handler: F) -> Self
    where
        F: Fn(T) -> R + Send + Sync + 'static,
    {
        let (job_sender, job_receiver) = mpsc::channel::<Option<T>>(); // Use Option<T>
        let (result_sender, result_receiver) = mpsc::channel::<R>();
        let job_receiver = Arc::new(Mutex::new(job_receiver));
        let result_sender = Arc::new(Mutex::new(result_sender));

        let mut workers = Vec::with_capacity(num_threads);

        // Wrap job_handler in Arc to allow sharing across threads
        let job_handler = Arc::new(job_handler);

        for _ in 0..num_threads {
            let job_receiver = Arc::clone(&job_receiver);
            let result_sender = Arc::clone(&result_sender);
            let job_handler = Arc::clone(&job_handler);

            let worker = thread::spawn(move || {
                while let Ok(job) = job_receiver.lock().unwrap().recv() {
                    if let Some(job) = job {
                        let result = job_handler(job);
                        result_sender.lock().unwrap().send(result).unwrap();
                    } else {
                        break; // Exit the loop on shutdown signal
                    }
                }
            });

            workers.push(worker);
        }

        Self {
            workers,
            sender: Some(job_sender),
            results: Arc::new(Mutex::new(result_receiver)),
        }
    }

    /// Sends a job to the thread pool.
    pub fn send(&self, job: T) {
        if let Some(sender) = &self.sender {
            sender.send(Some(job)).unwrap();
        }
    }

    /// Collects all results from the thread pool.
    pub fn collect_results(&self, num_jobs: usize) -> Vec<R> {
        let mut results = Vec::new();
        let receiver = self.results.lock().unwrap();

        for _ in 0..num_jobs {
            // Block until a result is available
            if let Ok(result) = receiver.recv() {
                results.push(result);
            }
        }

        results
    }
}

impl<T, R> Drop for ThreadPool<T, R> {
    fn drop(&mut self) {
        // Signal all threads to shut down
        if let Some(sender) = self.sender.take() {
            for _ in &self.workers {
                sender.send(None).unwrap(); // Send shutdown signal
            }
        }

        // Join all threads
        for worker in self.workers.drain(..) {
            worker.join().unwrap();
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_pool() {
        let pool = ThreadPool::new(4, |x: i32| x * 2);
        let num_jobs: usize = 10; // Explicitly specify the type as usize

        for i in 0..num_jobs {
            pool.send(i as i32); // Cast i to i32 since the closure expects i32
        }

        let results = pool.collect_results(num_jobs);
        assert_eq!(results.len(), num_jobs);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(*result, (i as i32) * 2);
        }
    }

    #[test]
    fn test_thread_pool_with_strings() {
        let pool = ThreadPool::new(4, |x: String| x.len());
        let num_jobs = 10;

        for i in 0..num_jobs {
            pool.send(format!("test{}", i));
        }

        let results = pool.collect_results(num_jobs);
        assert_eq!(results.len(), num_jobs);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(*result, format!("test{}", i).len());
        }
    }
}
//     }
