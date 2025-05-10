use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// A thread-safe randomizer that supports seeding and various random generation functions.
#[derive(Serialize, Clone)]
pub struct Randomizer {
    seed: u64, // Store the seed for serialization
    #[serde(skip)] // Skip serialization of the RNG itself
    rng: Arc<Mutex<StdRng>>, // Thread-safe random number generator
}

impl Randomizer {
    /// Creates a new randomizer with a given seed.
    pub(crate) fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_nanos() as u64
        });
        let rng = StdRng::seed_from_u64(seed);
        Self {
            seed,
            rng: Arc::new(Mutex::new(rng)),
        }
    }

    /// Generates a random permutation of integers from 0 to n-1.
    pub(crate) fn perm(&self, n: usize) -> Vec<usize> {
        let mut rng = self.rng.lock().unwrap();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut *rng);
        indices
    }

    pub fn float32(&self) -> f32 {
        let mut rng = self.rng.lock().unwrap();
        rng.random::<f32>()
    }
}

// Implement custom deserialization to recreate the RNG from the seed
impl<'de> Deserialize<'de> for Randomizer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RandomizerSeed {
            seed: u64,
        }

        let RandomizerSeed { seed } = RandomizerSeed::deserialize(deserializer)?;
        let rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            seed,
            rng: Arc::new(Mutex::new(rng)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Randomizer;
    use std::collections::HashSet;

    #[test]
    fn test_randomizer_with_seed() {
        // Initialize two randomizers with the same seed
        let seed = Some(42);
        let randomizer1 = Randomizer::new(seed);
        let randomizer2 = Randomizer::new(seed);

        // Check if the same permutation is generated
        let perm1 = randomizer1.perm(10);
        let perm2 = randomizer2.perm(10);
        assert_eq!(perm1, perm2, "Permutations should match for the same seed");

        // Check if the same float64 is generated
        let float1 = randomizer1.float32();
        let float2 = randomizer2.float32();
        assert_eq!(float1, float2, "Random floats should match for the same seed");
    }

    #[test]
    fn test_randomizer_without_seed() {
        // Initialize two randomizers without a seed
        let randomizer1 = Randomizer::new(None);
        let randomizer2 = Randomizer::new(None);

        // It's unlikely that the same permutation will be generated
        let perm1 = randomizer1.perm(10);
        let perm2 = randomizer2.perm(10);
        assert_ne!(perm1, perm2, "Permutations should differ without a seed");

        // It's unlikely that the same float64 will be generated
        let float1 = randomizer1.float32();
        let float2 = randomizer2.float32();
        assert_ne!(float1, float2, "Random floats should differ without a seed");
    }

    #[test]
    fn test_random_permutation() {
        let randomizer = Randomizer::new(Some(42));
        let n = 5;
        let perm = randomizer.perm(n);

        // Check if the permutation contains all unique values
        let unique_values: HashSet<_> = perm.iter().cloned().collect();
        assert_eq!(unique_values.len(), n, "Permutation should contain unique values");

        // Check if the permutation contains all integers from 0 to n-1
        for i in 0..n {
            assert!(unique_values.contains(&i), "Permutation should contain {}", i);
        }
    }

    #[test]
    fn test_random_float64() {
        let randomizer = Randomizer::new(Some(42));
        let random_value = randomizer.float32();

        // Check if the value is in the range [0, 1)
        assert!((0.0..1.0).contains(&random_value), "Random float should be in range [0, 1), got {}", random_value);
    }

    #[test]
    fn test_randomizer_deserialization() {
        let json = r#"{"seed": 55}"#;
        let randomizer: Randomizer = serde_json::from_str(json).expect("Failed to deserialize");
        assert_eq!(randomizer.seed, 55);
    }
}
