use super::{activations::Activation, matrix::Matrix};

pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f64,
    activation: Activation<'a>,
}

impl Network<'_> {
    pub fn new<'a>(layers: Vec<usize>, learning_rate: f64, activation: Activation<'a>) -> Network {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            learning_rate,
            activation,
        }
    }

    // Todo -> Read on activation functions e.g Sigmoid function :: 1/(1 + e^-x)
    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        // If we don't get the right amount of inputs
        if inputs.len() != self.layers[0] {
            panic!("Invalid number of inputs");
        }

        // current -> Keeps track of the current layer
        let mut current = Matrix::from(vec![inputs]).transpose();
        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .multiply(&current)
                .add(&self.biases[i])
                .map(self.activation.function);
            self.data.push(current.clone());
        }

        current.data[0].to_owned()
    }

    // Back propagation
    pub fn back_propagate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) {
        if targets.len() != self.layers[self.layers.len() - 1] {
            panic!("Invalida no. of targets");
        }

        let mut parsed = Matrix::from(vec![outputs]);
        // Get difference between each individual output neuron
        let mut errors = Matrix::from(vec![targets]).subtract(&parsed);
        let mut gradients = parsed.map(self.activation.function);

        // Go backwards
        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients
                .dot_multiply(&errors)
                .map(&|x| x * self.learning_rate);
        }
    }
}
