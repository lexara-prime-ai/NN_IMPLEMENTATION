use lib::{activations::SIGMOID, network::Network};

pub mod lib;

// Todo -> Review matrix library

fn main() {
    // the truth table of the XOR (exclusive or)

    // 0, 0 -> 0
    // 0, 1 -> 1
    // 1, 0 -> 1
    // 1, 1 -> 0
    /*
       [0.0, 0.0]	0.0
       [0.0, 1.0]	1.0
       [1.0, 0.0]	1.0
       [1.0, 1.0]	0.0

       [0.8, 0.7]	0.1
       [0.4, 0.9]	0.5
       [0.6, 0.3]	0.9

       [0.9, 0.8]	0.2
    */
    // Sample implementation
    // -> If only one of the inputs is 1,the output will be 1
    // -> If all inputs are 0 or 1, the output will be 0

    // Define data
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.2],
        vec![0.2, 0.5],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    // Expected outputs
    let targets = vec![
        vec![0.0],
        vec![0.3],
        vec![0.7],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    // Define network
    let mut network = Network::new(vec![2, 3, 1], 0.5, SIGMOID);

    // Training
    network.train(inputs, targets, 10000);

    println!("0 and 0: {:?}", network.feed_forward(vec![0.0, 0.0]));
    println!("0 and 1: {:?}", network.feed_forward(vec![0.0, 1.0]));
    println!("1 and 0: {:?}", network.feed_forward(vec![1.0, 0.0]));
    println!("1 and 1: {:?}", network.feed_forward(vec![1.0, 1.0]));
}
