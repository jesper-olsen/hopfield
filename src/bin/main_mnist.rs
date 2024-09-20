use hopfield::mnist;
use hopfield::HopfieldNet;

const LLEN: usize = 10;   // Number of labels
const Q: u8 = 2; // Quantization levels, e.g. 2, 4, 8
const D: u8 = (256usize/Q as usize) as u8;
const SS: usize = Q as usize * 28*28+LLEN+1; // state length
const DIR: &str = "MNIST/";

fn image_to_state(label: &[u8], im: &[u8]) -> Vec<u8> {
    // one-hot encode intensity
    let p: Vec<u8> = im.iter()
        .flat_map(|&byte| {
            let bin = byte / D; // D bins
            (0..Q).map(move |i| if i == bin { 1 } else { 0 })
        })
        .collect();

    let mut x = Vec::with_capacity(label.len()+p.len()+1);
    x.push(1); // bias
    x.extend_from_slice(label);
    x.extend_from_slice(&p);
    x
}

fn state_to_image(state: &[u8], label_len: usize) -> Vec<u8> {
    let state = &state[label_len + 1..]; // Skip bias and label

    state
        .chunks(Q as usize) // Each pixel is represented by D one-hot values
        .map(|bin| {
            // Decode the one-hot encoding back into intensity
            bin.iter()
                .enumerate()
                .find(|(_, &v)| v == 1) // Find which bit is set to 1
                .map(|(i, v)| (i as u8) * D) // Map back to intensity (0 to 255, D bins)
                .unwrap_or(0) // Default to 0 if no bit is set (edge case)
        })
        .collect()
}

fn generate_one_hot_state_vectors(num_labels: usize, state_length: usize) -> Vec<Vec<u8>> {
    let mut state_vectors = vec![vec![0; state_length]; num_labels];
    for i in 0..num_labels {
        state_vectors[i][i] = 1;
    }
    state_vectors
}

fn mnist_train(nepochs: usize) {
    let cb = generate_one_hot_state_vectors(10, LLEN);
    for (i, v) in cb.iter().enumerate() {
        println!("{i}: {v:?}")
    }

    //for s in ["train", "t10k"] {
    let fname = format!("{DIR}train-labels.idx1-ubyte");
    let labels = mnist::read_labels(&fname).unwrap();
    println!("Read {} labels", labels.len());

    let fname = format!("{DIR}train-images.idx3-ubyte");
    let images = mnist::read_images(&fname).unwrap();
    let mut net = HopfieldNet::<SS>::new();
    for j in 0..nepochs {
        for (i, (im, lab)) in images.iter().zip(labels.iter()).enumerate() {
            //mnist::plot_image(im, 28,28,*lab);
            let x = image_to_state(&cb[*lab as usize], im);
            let _ = net.perceptron_conv_procedure(&x);
            if i % 100 == 0 {
                println!("{j},{i}");
            }
        }
        let fname = format!("hop{j}.json");
        net.save_json(&fname).expect("failed to save");
    }
    mnist_test(&cb, &mut net)
}

fn predict(cb: &[Vec<u8>], x: &[u8]) -> usize {
        let mut mind = LLEN + 1;
        let mut mini = 0;
        for (i, v) in cb.iter().enumerate() {
            let d: usize = v
                .iter()
                .zip(&x[1..v.len()])
                .map(|(x, y)| if x == y { 0 } else { 1 })
                .sum();
            //println!("lab {i} d {d}");
            if d < mind {
                mind = d;
                mini = i;
            }
        }
        mini
}

fn mnist_test(cb: &[Vec<u8>], net: &HopfieldNet<SS>) {
    let fname = format!("{DIR}t10k-labels.idx1-ubyte");
    let labels = mnist::read_labels(&fname).unwrap();
    println!("Read {} labels", labels.len());

    let fname = format!("{DIR}t10k-images.idx3-ubyte");
    let images = mnist::read_images(&fname).unwrap();

    let mut correct = 0;
    let mut n = 0;
    for (_i, (im, lab)) in images.iter().zip(labels.iter()).enumerate() {
        //mnist::plot_image(im, 28,28,*lab);
        let mut x = image_to_state(&[0;10], im);

        let mut g0 = net.goodness(&x);
        println!("Goodness: {g0}");
        let mut mini;
        loop {
            net.step(&mut x);
            let g1 = net.goodness(&x);
            mini = predict(&cb,&x);
            println!("Goodness: {g1}, lab: {lab} prediction: {mini}");
            if g1 == g0 {
                break;
            }
            g0 = g1
        }

        n += 1;
        if *lab as usize == mini {
            correct += 1;
        }
        println!("correct {correct}/{n}");
        
        let imp=state_to_image(&x, 10);
        println!("im len: {}, imp len: {} x len {}", im.len(), imp.len(), x.len());
        mnist::plot_image(&im, 28,28,*lab);
        mnist::plot_image(&imp, 28,28,*lab);
    }
}

fn main() {
    mnist_train(1);

    //let fname = format!("WEIGHTS/hop0.json");
    //let mut net = HopfieldNet::<SS>::load_json(&fname).expect("Failed to load Hopfield network");
    //let cb = generate_one_hot_state_vectors(10, LLEN);
    //mnist_test(&cb, &mut net);
}
