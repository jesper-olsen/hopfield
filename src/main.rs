use std::collections::HashSet;
pub mod hopfield;
pub mod mnist;
use hopfield::HopfieldNet;
use stmc_rs::marsaglia::Marsaglia;

const LLEN: usize = 10;
const SS: usize = 8*28*28+LLEN+1;
const DIR: &str = "MNIST/";

fn state2u64(state: &[u8]) -> u64 {
    state
        .iter()
        .skip(1) // bias term
        .enumerate()
        .fold(0, |b, (i, &x)| b | ((x as u64) << i))
}

fn u64_to_state(a: u64) -> Vec<u8> {
    std::iter::once(1)  // bias term 1
        .chain(
            //(0..64).rev().map(move |i| if a & (1 << i) != 0 { 1 } else { 0 })
            (0..64).map(move |i| if a & (1 << i) != 0 { 1 } else { 0 })
        )
        .collect()
}

fn image_to_state(label: &[u8], im: &[u8]) -> Vec<u8> {
    // one-hot encode intensity
    let p: Vec<u8> = im.iter()
        .flat_map(|&byte| {
            let bin = (byte / 32) as usize; // 8 bins
            (0..8).map(move |i| if i == bin { 1 } else { 0 })
        })
        .collect();

    let mut x = Vec::with_capacity(label.len()+p.len()+1);
    x.push(1); // bias
    x.extend_from_slice(label);
    x.extend_from_slice(&p);
    x
}

fn generate_unique_state_vectors(num_labels: usize, state_length: usize) -> Vec<Vec<u8>> {
    let mut rng = Marsaglia::new(12, 34, 56, 78);
    let mut state_vectors = HashSet::new();

    while state_vectors.len() < num_labels {
        let mut state = Vec::with_capacity(state_length);
        for _ in 0..state_length {
            let flag = if rng.uni() > 0.5 { 1 } else { 0 };
            state.push(flag);
        }

        if state_vectors.insert(state) {
            // Ensure uniqueness
        }
    }
    state_vectors.into_iter().collect()
}

fn generate_one_hot_state_vectors(num_labels: usize, state_length: usize) -> Vec<Vec<u8>> {
    let mut state_vectors = vec![vec![0; state_length]; num_labels];
    for i in 0..num_labels {
        state_vectors[i][i] = 1;
    }
    state_vectors
}

fn mnist_train(nepochs: usize) {
    //let cb = generate_unique_state_vectors(10,LLEN);
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
            let change = net.perceptron_conv_procedure(&x);
            if i % 100 == 0 {
                println!("{j},{i}");
            }
        }
        let fname = format!("hop{j}.json");
        //net.save_json(&fname).expect("failed to save");
        net.save_json(&fname).expect("failed to save");
        let fname = format!("hop_weights{j}.bin");
        net.save_weights(&fname).expect("failed to save");
    }
    mnist_test(&mut net)
}

fn predict(net: &HopfieldNet<SS>, cb: &[Vec<u8>], x: &[u8]) -> usize {
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

fn mnist_test(net: &HopfieldNet<SS>) {
    let cb = generate_one_hot_state_vectors(10, LLEN);
    let dir = "MNIST/";
    let fname = format!("{DIR}t10k-labels.idx1-ubyte");
    let labels = mnist::read_labels(&fname).unwrap();
    println!("Read {} labels", labels.len());

    let fname = format!("{DIR}t10k-images.idx3-ubyte");
    let images = mnist::read_images(&fname).unwrap();

    let mut correct = 0;
    let mut n = 0;
    for (i, (im, lab)) in images.iter().zip(labels.iter()).enumerate() {
        //mnist::plot_image(im, 28,28,*lab);
        let mut x = image_to_state(&[0;10], im);

        let mut g0 = net.goodness(&x);
        println!("Goodness: {g0}");
        let mut mini;
        loop {
            net.step(&mut x);
            let g1 = net.goodness(&x);
            mini = predict(net,&cb,&x);
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
    }
}

fn hop_font8x8() {
    let mut net = HopfieldNet::<65>::new();
    if true {
        for i in 0x61..0x64 {
            let b = font8x8::unicode2bitmap(i);
            font8x8::display(b);
            let v = u64_to_state(b);
            net.hopfield_storage_rule(&v);
        }
    } else {
        for _ in 0..3 {
            for i in 0x61..0x64 {
                let b = font8x8::unicode2bitmap(i);
                font8x8::display(b);
                let v = u64_to_state(b);
                for _ in 0..10 {
                    let change = net.perceptron_conv_procedure(&v);
                }
            }
        }
    }
    let mask0: u64 = 0xFFFFFFFFFFFFFFFF;
    let mask1: u64 = 0xFFFFFFFF00000000;
    let mask2: u64 = 0x00000000FFFFFFFF;
    for (label, mask) in [("none", mask0), ("upper", mask1), ("lower", mask2)] {
        for i in 0x61..=0x66 {
            let ch = char::from_u32(i as u32).unwrap();
            println!("Initialising with {ch} - mask: {label}");
            let b = mask & font8x8::unicode2bitmap(i);
            font8x8::display(b);
            let mut v = u64_to_state(b);
            println!("Goodness: {}", net.goodness(&v));
            for _ in 0..3 {
                net.step(&mut v);
                println!("Goodness: {}", net.goodness(&v));
            }
            let b = state2u64(&v);
            font8x8::display(b);
        }
    }
}

fn main() {
    //hop_font8x8();
    mnist_train(3);
    //let mut hnet = hopfield::load_json("hop0.json").expect("Failed to load model");

//    let fname = format!("hop_weights2.bin");
//    let mut net = HopfieldNet::<SS>::new();
//    net.load_weights(&fname).expect("failed to save");
//    mnist_test(&mut net);
}
