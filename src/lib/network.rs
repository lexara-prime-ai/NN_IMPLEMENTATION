use super::matrix::Matrix;

pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
}

impl Network {
    pub fn new(layers: Vec<usize>) -> Network {
        let mut weights = vec![];
        let mut biases = vec![];
    }
}
