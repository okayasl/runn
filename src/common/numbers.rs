use rand::{Rng, SeedableRng};

pub trait Numbers {
    fn floats(&self) -> Vec<f32>;
    fn ints(&self) -> Vec<usize>;
}

/// Generates a sequence of numbers in a given range with a specified increment.
/// The sequence starts from the lower limit and goes up to the upper limit.
/// The increment must be greater than 0.
/// The generated numbers are rounded to 6 decimal places.
pub struct SequentialNumbers {
    lower_limit: f32,
    upper_limit: f32,
    increment: f32,
}

impl SequentialNumbers {
    pub fn new() -> Self {
        SequentialNumbers {
            lower_limit: 0.0,
            upper_limit: 0.0,
            increment: 0.0,
        }
    }

    pub fn lower_limit(mut self, ll: f32) -> Self {
        self.lower_limit = ll;
        self
    }

    pub fn upper_limit(mut self, ul: f32) -> Self {
        self.upper_limit = ul;
        self
    }

    pub fn increment(mut self, inc: f32) -> Self {
        self.increment = inc;
        self
    }
    fn check_params(&self) {
        if self.lower_limit < 0.0 {
            panic!("lower limit must be greater than or equal to 0");
        }
        if self.upper_limit < 0.0 {
            panic!("upper limit must be greater than or equal to 0");
        }
        if self.lower_limit > self.upper_limit {
            panic!("lower limit is greater than upper limit");
        }
        if self.increment <= 0.0 {
            panic!("increment must be greater than 0");
        }
    }
}

impl Numbers for SequentialNumbers {
    fn floats(&self) -> Vec<f32> {
        self.check_params();
        let mut values = Vec::new();
        let mut n = 0;
        loop {
            let mut v = self.lower_limit + self.increment * n as f32;
            v = (v * 1_000_000.0).round() / 1_000_000.0;
            if v > self.upper_limit + f32::EPSILON {
                break;
            }
            values.push(v);
            n += 1;
        }
        values
    }

    fn ints(&self) -> Vec<usize> {
        self.check_params();
        let mut values = Vec::new();
        let mut v = self.lower_limit as usize;
        while v <= self.upper_limit as usize {
            values.push(v);
            v += self.increment as usize;
        }
        values
    }
}

/// Generates a sequence of random numbers in a given range.
/// The sequence starts from the lower limit and goes up to the upper limit.
/// The size of the sequence must be greater than 0.
/// The generated numbers are unique and randomly ordered.
pub struct RandomNumbers {
    lower_limit: f32,
    upper_limit: f32,
    size: usize,
    seed: u64,
}

impl RandomNumbers {
    pub fn new() -> Self {
        RandomNumbers {
            lower_limit: 0.0,
            upper_limit: 0.0,
            size: 0,
            seed: rand::thread_rng().gen::<u64>(),
        }
    }

    pub fn lower_limit(mut self, ll: f32) -> Self {
        self.lower_limit = ll;
        self
    }

    pub fn upper_limit(mut self, ul: f32) -> Self {
        self.upper_limit = ul;
        self
    }

    pub fn size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn check_params(&self) {
        if self.lower_limit < 0.0 {
            panic!("lower limit must be greater than or equal to 0");
        }
        if self.upper_limit < 0.0 {
            panic!("upper limit must be greater than or equal to 0");
        }
        if self.lower_limit > self.upper_limit {
            panic!("lower limit is greater than upper limit");
        }
        if self.size == 0 {
            panic!("size must be greater than 0");
        }
    }
}

impl Numbers for RandomNumbers {
    fn floats(&self) -> Vec<f32> {
        self.check_params();
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        (0..self.size)
            .map(|_| rng.gen_range(self.lower_limit..self.upper_limit))
            .collect()
    }

    fn ints(&self) -> Vec<usize> {
        self.check_params();
        let possible = (self.upper_limit as usize) - (self.lower_limit as usize) + 1;
        if self.size > possible {
            panic!("Requested more unique numbers than possible in range");
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        while result.len() < self.size {
            let num = rng.gen_range(self.lower_limit as usize..=self.upper_limit as usize);
            if seen.insert(num) {
                result.push(num);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_floats_basic() {
        let numbers = SequentialNumbers::new()
            .lower_limit(0.0)
            .upper_limit(2.0)
            .increment(1.0);

        let expected = vec![0.0, 1.0, 2.0];
        assert!(numbers
            .floats()
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn sequential_floats_basic2() {
        let numbers = SequentialNumbers::new()
            .lower_limit(0.0020)
            .upper_limit(0.0030)
            .increment(0.0005);

        let expected = vec![0.0020, 0.0025, 0.0030];
        assert!(numbers
            .floats()
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn sequential_ints_basic() {
        let numbers = SequentialNumbers::new()
            .lower_limit(1.0)
            .upper_limit(5.0)
            .increment(2.0);

        let expected = vec![1, 3, 5];
        assert_eq!(numbers.ints(), expected);
    }

    #[test]
    #[should_panic(expected = "lower limit must be greater than or equal to 0")]
    fn sequential_negative_lower_limit_panics() {
        SequentialNumbers::new()
            .lower_limit(-1.0)
            .upper_limit(5.0)
            .increment(1.0)
            .floats();
    }

    #[test]
    #[should_panic(expected = "increment must be greater than 0")]
    fn sequential_zero_increment_panics() {
        SequentialNumbers::new()
            .lower_limit(0.0)
            .upper_limit(5.0)
            .increment(0.0)
            .floats();
    }

    #[test]
    fn random_floats_with_fixed_seed() {
        let numbers = RandomNumbers::new().lower_limit(0.0).upper_limit(1.0).size(5).seed(42);

        let floats = numbers.floats();
        assert_eq!(floats.len(), 5);
        for &v in &floats {
            assert!(v >= 0.0 && v < 1.0);
        }

        // Same seed = same result
        let numbers2 = RandomNumbers::new().lower_limit(0.0).upper_limit(1.0).size(5).seed(42);

        assert_eq!(floats, numbers2.floats());
    }

    #[test]
    fn random_ints_with_fixed_seed() {
        let numbers = RandomNumbers::new()
            .lower_limit(0.0)
            .upper_limit(10.0)
            .size(5)
            .seed(123);

        let ints = numbers.ints();
        assert_eq!(ints.len(), 5);

        for &v in &ints {
            assert!(v <= 10);
        }

        let unique: std::collections::HashSet<_> = ints.iter().copied().collect();
        assert_eq!(unique.len(), ints.len()); // all unique
    }

    #[test]
    #[should_panic(expected = "size must be greater than 0")]
    fn random_zero_size_panics() {
        RandomNumbers::new().lower_limit(0.0).upper_limit(10.0).size(0).floats();
    }

    #[test]
    #[should_panic(expected = "lower limit is greater than upper limit")]
    fn random_lower_greater_than_upper_panics() {
        RandomNumbers::new().lower_limit(10.0).upper_limit(5.0).size(5).floats();
    }
}
