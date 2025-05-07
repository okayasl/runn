use std::fmt::Write;

use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::util::find_max_index_in_row;

use super::matrix::DMat;

/// A mode flag so we know whether to do classification or regression
pub enum CompareMode {
    Classification,
    Regression,
}

/// Pretty print three matrices side-by-side as an ASCII-art comparison and return as a String.
pub fn pretty_compare_matrices(input: &DMat, target: &DMat, prediction: &DMat, mode: CompareMode) -> String {
    let rows = input.rows();
    let mut buf = String::new();

    // character width per entry (including space)
    let char_length = 8; // each formatted number uses 7 chars + 1 space

    // Calculate block widths (borders + columns*char_length)
    let input_w = input.cols() * char_length + 2;
    let target_w = target.cols() * char_length + 2;
    let pred_w = prediction.cols() * char_length + 2;

    // Header: center labels above each matrix block with dynamic widths
    writeln!(
        buf,
        "\n{:^input_w$}  {:^target_w$}  {:^pred_w$}",
        "Input",
        "Target",
        "Prediction",
        input_w = input_w,
        target_w = target_w,
        pred_w = pred_w
    )
    .unwrap();

    if rows == 0 {
        return buf;
    }

    // Helper to format one row of any matrix with given border chars
    let format_row = |mat: &DMat, i: usize, left: char, right: char| {
        let mut s = String::new();
        s.push(left);
        for j in 0..mat.cols() {
            write!(s, "{:>7.2} ", mat.at(i, j)).unwrap();
        }
        s.push(right);
        s
    };

    // Helper to compute regression error
    let row_error = |i: usize| {
        let mut err = 0.0;
        for j in 0..target.cols() {
            err += (prediction.at(i, j) - target.at(i, j)).abs();
        }
        format!(">{:7.2}", err)
    };

    // Iterate through rows, choosing border style per position
    for i in 0..rows {
        let (l, r) = if rows == 1 {
            ('[', ']')
        } else if i == 0 {
            ('⎡', '⎤')
        } else if i + 1 == rows {
            ('⎣', '⎦')
        } else {
            ('⎢', '⎥')
        };

        let inp = format_row(input, i, l, r);
        let tgt = format_row(target, i, l, r);
        let pred = format_row(prediction, i, l, r);

        // writeln!(buf, "{}  {}{}  {}", inp, tgt, marker(i), pred).unwrap();

        match mode {
            CompareMode::Classification => {
                // classification marker between target and prediction
                let mark = if find_max_index_in_row(target, i) != find_max_index_in_row(prediction, i) {
                    " <>"
                } else {
                    "   "
                };
                writeln!(buf, "{}  {}{}  {}", inp, tgt, mark, pred).unwrap();
            }
            CompareMode::Regression => {
                // regression error at far end
                let err_str = row_error(i);
                writeln!(buf, "{}  {}  {}  {}", inp, tgt, pred, err_str).unwrap();
            }
        }
    }

    buf
}

pub fn one_hot_encode(targets: &DMat) -> DMat {
    // Find the unique values in the target matrix to determine the number of classes
    let mut unique_values: Vec<f32> = Vec::new();

    for i in 0..targets.rows() {
        for j in 0..targets.cols() {
            let target_value = targets.at(i, j);
            if !unique_values.contains(&target_value) {
                unique_values.push(target_value);
            }
        }
    }

    // Sort the unique values and create a mapping for each class
    unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let num_classes = unique_values.len();

    // Initialize the one-hot encoded matrix with `num_classes` columns for each target
    let mut one_hot_targets = DMat::zeros(targets.rows(), targets.cols() * num_classes);

    for i in 0..targets.rows() {
        for j in 0..targets.cols() {
            let target_value = targets.at(i, j);
            // Find the index of the class (which class it corresponds to)
            let class_index = unique_values.iter().position(|&x| x == target_value).unwrap();
            one_hot_targets.set(i, j * num_classes + class_index, 1.0); // Set 1 in the correct column
        }
    }

    one_hot_targets
}

/// Random split for matrix inputs and multi-columbn targets
///
/// # Arguments
/// * `inputs` - DenseMatrix of shape (n_samples, n_features)
/// * `targets` - DenseMatrix of shape (n_samples, n_target_features)
/// * `validation_ratio` - e.g. 0.2 means 20% validation, 80% training
/// * `seed` - RNG seed for reproducibility
///
/// # Returns
/// `(train_inputs, train_targets, val_inputs, val_targets)`
pub fn random_split(inputs: &DMat, targets: &DMat, validation_ratio: f32, seed: u64) -> (DMat, DMat, DMat, DMat) {
    let n = inputs.rows();
    assert_eq!(n, targets.rows(), "Row count of inputs and targets must match");
    assert!((0.0..=1.0).contains(&validation_ratio), "validation_ratio must be in [0,1]");

    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    let val_count = (n as f32 * validation_ratio).round() as usize;
    let (val_idx, train_idx) = indices.split_at(val_count);

    let input_cols = inputs.cols();
    let target_cols = targets.cols();

    let mut train_inputs_data = Vec::with_capacity(train_idx.len() * input_cols);
    let mut train_targets_data = Vec::with_capacity(train_idx.len() * target_cols);
    let mut val_inputs_data = Vec::with_capacity(val_idx.len() * input_cols);
    let mut val_targets_data = Vec::with_capacity(val_idx.len() * target_cols);

    for &i in train_idx {
        train_inputs_data.extend(inputs.get_row(i));
        train_targets_data.extend(targets.get_row(i));
    }

    for &i in val_idx {
        val_inputs_data.extend(inputs.get_row(i));
        val_targets_data.extend(targets.get_row(i));
    }

    let train_inputs = DMat::new(train_idx.len(), input_cols, &train_inputs_data);
    let train_targets = DMat::new(train_idx.len(), target_cols, &train_targets_data);
    let val_inputs = DMat::new(val_idx.len(), input_cols, &val_inputs_data);
    let val_targets = DMat::new(val_idx.len(), target_cols, &val_targets_data);

    (train_inputs, train_targets, val_inputs, val_targets)
}

/// Stratified split for matrix inputs and single-column targets
///
/// # Arguments
/// * `inputs` - DenseMatrix of shape (n_samples, n_features)
/// * `targets` - DenseMatrix of shape (n_samples, 1) containing class labels (e.g. 0..10)
/// * `validation_ratio` - e.g. 0.2 means 20% validation, 80% training
/// * `seed` - RNG seed for reproducibility
///
/// # Returns
/// `(train_inputs, train_targets, val_inputs, val_targets)`
pub fn stratified_split(inputs: &DMat, targets: &DMat, validation_ratio: f32, seed: u64) -> (DMat, DMat, DMat, DMat) {
    let n = inputs.rows();
    assert_eq!(n, targets.rows(), "Row count of inputs and targets must match");
    assert!(targets.cols() == 1, "Targets must be a single-column matrix");
    assert!((0.0..=1.0).contains(&validation_ratio), "validation_ratio must be in [0,1]");

    // Group sample indices by class label value using a Vec of (label, indices)
    let mut buckets: Vec<(f32, Vec<usize>)> = Vec::new();
    for i in 0..n {
        let label = targets.at(i, 0);
        if let Some((_, ref mut vec)) = buckets.iter_mut().find(|(l, _)| *l == label) {
            vec.push(i);
        } else {
            buckets.push((label, vec![i]));
        }
    }

    // Sort buckets by label for deterministic ordering
    buckets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Prepare RNG
    let mut rng = StdRng::seed_from_u64(seed);
    let mut train_idx: Vec<usize> = Vec::new();
    let mut val_idx: Vec<usize> = Vec::new();

    // Shuffle and split each bucket
    for (_label, mut idxs) in buckets {
        idxs.shuffle(&mut rng);
        let val_count = ((idxs.len() as f32) * validation_ratio).round() as usize;
        let (v, t) = idxs.split_at(val_count);
        val_idx.extend_from_slice(v);
        train_idx.extend_from_slice(t);
    }

    let n_features = inputs.cols();
    // build flat data for matrices
    let mut train_flat = Vec::with_capacity(train_idx.len() * n_features);
    let mut train_t_flat = Vec::with_capacity(train_idx.len());
    for &i in &train_idx {
        train_flat.extend(inputs.get_row(i));
        train_t_flat.push(targets.at(i, 0));
    }
    let mut val_flat = Vec::with_capacity(val_idx.len() * n_features);
    let mut val_t_flat = Vec::with_capacity(val_idx.len());
    for &i in &val_idx {
        val_flat.extend(inputs.get_row(i));
        val_t_flat.push(targets.at(i, 0));
    }

    let train_inputs = DMat::new(train_idx.len(), n_features, &train_flat);
    let train_targets = DMat::new(train_idx.len(), 1, &train_t_flat);
    let val_inputs = DMat::new(val_idx.len(), n_features, &val_flat);
    let val_targets = DMat::new(val_idx.len(), 1, &val_t_flat);

    (train_inputs, train_targets, val_inputs, val_targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_split_shapes() {
        // 6 samples, 2 features
        let flat_in = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let inputs = DMat::new(6, 2, &flat_in);
        // targets: three classes 0,1,2 twice each
        let t = vec![0., 0., 1., 1., 2., 2.];
        let targets = DMat::new(6, 1, &t);
        let (ti, tt, vi, vt) = stratified_split(&inputs, &targets, 0.5, 1234);
        // 50% => one of each class in val => 3 val, 3 train
        assert_eq!(ti.rows(), 3);
        assert_eq!(vi.rows(), 3);
        assert_eq!(tt.rows(), 3);
        assert_eq!(vt.rows(), 3);
        assert_eq!(ti.cols(), 2);
        assert_eq!(vi.cols(), 2);
    }

    #[test]
    fn test_matrix_split_stratify() {
        let flat_in: Vec<f32> = (0..20).map(|x| x as f32).collect();
        let inputs = DMat::new(10, 2, &flat_in); // 10 samples
                                                 // classes: 0 for first 5, 1 for next 5
        let mut t = Vec::new();
        t.extend(vec![0.; 5]);
        t.extend(vec![1.; 5]);
        let targets = DMat::new(10, 1, &t);
        let (_ti, _tt, _vi, vt) = stratified_split(&inputs, &targets, 0.4, 42);
        // 40% of each class => 2 from class0 and 2 from class1 in val
        let mut count0 = 0;
        let mut count1 = 0;
        for i in 0..vt.rows() {
            match vt.at(i, 0) as usize {
                0 => count0 += 1,
                1 => count1 += 1,
                _ => {}
            }
        }
        assert_eq!(count0, 2);
        assert_eq!(count1, 2);
    }

    #[test]
    fn test_matrix_reproducibility() {
        // 8 samples, 3 features, classes 0/1 repeated
        let flat_in: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let inputs = DMat::new(8, 3, &flat_in);
        let mut t = Vec::new();
        for i in 0..8 {
            t.push((i % 4) as f32);
        }
        let targets = DMat::new(8, 1, &t);
        let seed = 2025;
        let (a_i1, a_t1, v_i1, v_t1) = stratified_split(&inputs, &targets, 0.25, seed);
        let (a_i2, a_t2, v_i2, v_t2) = stratified_split(&inputs, &targets, 0.25, seed);
        // compare shapes
        assert_eq!(a_i1.rows(), a_i2.rows());
        assert_eq!(v_i1.rows(), v_i2.rows());
        // compare content row-wise
        for i in 0..a_i1.rows() {
            assert_eq!(a_i1.get_row(i), a_i2.get_row(i));
            assert_eq!(a_t1.at(i, 0), a_t2.at(i, 0));
        }
        for i in 0..v_i1.rows() {
            assert_eq!(v_i1.get_row(i), v_i2.get_row(i));
            assert_eq!(v_t1.at(i, 0), v_t2.at(i, 0));
        }
    }

    fn create_dummy_matrix(rows: usize, cols: usize) -> DMat {
        let data: Vec<f32> = (0..rows * cols).map(|x| x as f32).collect();
        DMat::new(rows, cols, &data)
    }

    #[test]
    fn test_random_split_shapes() {
        let inputs = create_dummy_matrix(10, 4);
        let targets = create_dummy_matrix(10, 2);

        let (train_inputs, train_targets, val_inputs, val_targets) = random_split(&inputs, &targets, 0.2, 42);

        assert_eq!(train_inputs.rows(), 8);
        assert_eq!(train_inputs.cols(), 4);
        assert_eq!(train_targets.rows(), 8);
        assert_eq!(train_targets.cols(), 2);
        assert_eq!(val_inputs.rows(), 2);
        assert_eq!(val_inputs.cols(), 4);
        assert_eq!(val_targets.rows(), 2);
        assert_eq!(val_targets.cols(), 2);
    }

    #[test]
    fn test_random_split_reproducibility() {
        let inputs = create_dummy_matrix(20, 3);
        let targets = create_dummy_matrix(20, 2);

        let (train1, target1, val1, valt1) = random_split(&inputs, &targets, 0.3, 123);
        let (train2, target2, val2, valt2) = random_split(&inputs, &targets, 0.3, 123);

        assert_eq!(train1.at(10, 2), train2.at(10, 2));
        assert_eq!(target1.at(3, 0), target2.at(3, 0));
        assert_eq!(val1.at(4, 2), val2.at(4, 2));
        assert_eq!(valt1.at(5, 1), valt2.at(5, 1));
    }

    #[test]
    #[should_panic(expected = "Row count of inputs and targets must match")]
    fn test_random_split_mismatched_rows_should_panic() {
        let inputs = create_dummy_matrix(5, 2);
        let targets = create_dummy_matrix(4, 1); // Mismatched row count

        let _ = random_split(&inputs, &targets, 0.5, 1);
    }

    #[test]
    #[should_panic(expected = "validation_ratio must be in [0,1]")]
    fn test_random_split_invalid_ratio_should_panic() {
        let inputs = create_dummy_matrix(5, 2);
        let targets = create_dummy_matrix(5, 1);

        let _ = random_split(&inputs, &targets, 1.5, 1); // Invalid ratio
    }

    #[test]
    fn test_random_split_empty_matrix() {
        let inputs = create_dummy_matrix(0, 2);
        let targets = create_dummy_matrix(0, 1);

        let (train_inputs, train_targets, val_inputs, val_targets) = random_split(&inputs, &targets, 0.3, 99);

        assert_eq!(train_inputs.rows(), 0);
        assert_eq!(train_inputs.cols(), 2);
        assert_eq!(train_targets.rows(), 0);
        assert_eq!(train_targets.cols(), 1);
        assert_eq!(val_inputs.rows(), 0);
        assert_eq!(val_inputs.cols(), 2);
        assert_eq!(val_targets.rows(), 0);
        assert_eq!(val_targets.cols(), 1);
    }
}
