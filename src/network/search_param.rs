use rand::{Rng, SeedableRng};

pub trait Parameters {
    fn float_parameters(&self) -> Vec<f32>;
    fn int_parameters(&self) -> Vec<usize>;
}

pub struct RangeParameters {
    lower_limit: f32,
    upper_limit: f32,
    increment: f32,
}

impl RangeParameters {
    pub fn new() -> Self {
        RangeParameters {
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
}

impl Parameters for RangeParameters {
    fn float_parameters(&self) -> Vec<f32> {
        let mut values = Vec::new();
        let mut v = self.lower_limit;
        while v <= self.upper_limit {
            values.push(v);
            v += self.increment;
        }
        values
    }

    fn int_parameters(&self) -> Vec<usize> {
        let mut values = Vec::new();
        let mut v = self.lower_limit as usize;
        while v <= self.upper_limit as usize {
            values.push(v);
            v += self.increment as usize;
        }
        values
    }
}

pub struct RandomParameters {
    lower_limit: f32,
    upper_limit: f32,
    size: usize,
    seed: u64,
}

impl RandomParameters {
    pub fn new() -> Self {
        RandomParameters {
            lower_limit: 0.0,
            upper_limit: 0.0,
            size: 0,
            seed: 0,
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

impl Parameters for RandomParameters {
    fn float_parameters(&self) -> Vec<f32> {
        self.check_params();
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let mut seen = Vec::new();
        let mut result = Vec::new();

        while result.len() < self.size {
            let num = rng.gen_range(self.lower_limit..self.upper_limit);
            if !seen.contains(&num) {
                seen.push(num);
                result.push(num);
            }
        }

        result
    }

    fn int_parameters(&self) -> Vec<usize> {
        self.check_params();
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
