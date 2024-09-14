use std::collections::HashSet;
pub mod mnist;
pub mod hopfield;
use hopfield::HopfieldNet;
use stmc_rs::marsaglia::Marsaglia;

fn state2u64(state: &[u8]) -> u64 {
    state
        .iter()
        .enumerate()
        .fold(0, |b, (i, &x)| b | ((x as u64) << i))
}

fn bslice_to_state(a: &[u8]) -> Vec<u8> {
    a.iter()
        .flat_map(|&byte| {
            (0..8).rev().map(move |i| if byte & (1 << i) != 0 { 1 } else { 0 })
        })
        .collect()
}

fn u64_to_state(b: u64) -> Vec<u8> {
    bslice_to_state(&b.to_le_bytes())
}

fn generate_unique_state_vectors(num_labels: usize, state_length: usize) -> Vec<Vec<u8>> {
    let mut rng = Marsaglia::new(12, 34, 56, 78);
    let mut state_vectors = HashSet::new();

    while state_vectors.len() < num_labels {
        let mut state = Vec::with_capacity(state_length);
        for _ in 0..state_length {
            let flag = if rng.uni()>0.5 {1} else {0};
            state.push(flag);
        }

        if state_vectors.insert(state) {
            // Ensure uniqueness
        }
    }
    state_vectors.into_iter().collect()
}

fn generate_one_hot_state_vectors(num_labels: usize, state_length: usize) -> Vec<Vec<u8>> {
    let mut state_vectors = vec![vec![0;state_length];num_labels];
    for i in 0..num_labels {
        state_vectors[i][i]=1;
    }
    state_vectors
}

fn mnist_classification() {
    const LLEN: usize = 10;
    //let cb = generate_unique_state_vectors(10,LLEN);
    let cb = generate_one_hot_state_vectors(10,LLEN);
    for (i,v) in cb.iter().enumerate() {
        println!("{i}: {v:?}")
    }

    let dir = "MNIST/";
    //for s in ["train", "t10k"] {
    let fname = format!("{dir}train-labels.idx1-ubyte");
    let labels = mnist::read_labels(&fname).unwrap();
    println!("Read {} labels", labels.len());

    let fname = format!("{dir}train-images.idx3-ubyte");
    let images = mnist::read_images(&fname).unwrap();
    let mut net = HopfieldNet::new(&[0; 8*28*28+LLEN]);
    for j in 0..3 {
        for (i,(im,lab)) in images.iter().zip(labels.iter()).enumerate() {
            //mnist::plot_image(im, 28,28,*lab);
            let v = bslice_to_state(im);
            let mut x = vec![0;LLEN+v.len()];
            for (i,b) in cb[*lab as usize].iter().enumerate() {
                x[i]=*b;
            }
            for (i,b) in v.iter().enumerate() {
                x[LLEN+i]=*b;
            }
            net.set_state(&x);
            let change = net.perceptron_conv_procedure();
            println!("{j},{i}");
        }
    }

    let fname = format!("{dir}t10k-labels.idx1-ubyte");
    let labels = mnist::read_labels(&fname).unwrap();
    println!("Read {} labels", labels.len());

    let fname = format!("{dir}t10k-images.idx3-ubyte");
    let images = mnist::read_images(&fname).unwrap();

    let mut correct=0;
    let mut n=0;
    for (i,(im,lab)) in images.iter().zip(labels.iter()).enumerate() {
        //mnist::plot_image(im, 28,28,*lab);
        let v = bslice_to_state(im);
        let mut x = vec![0;LLEN+v.len()];
        //for (i,b) in cb[*lab as usize].iter().enumerate() {
        //    x[i]=*b;
        //}
        for (i,b) in v.iter().enumerate() {
            x[LLEN+i]=*b;
        }
        net.set_state(&x);
 
        let mut g0 = net.goodness();
        println!("Goodness: {g0}");
        loop {
           net.step();
           let g1 = net.goodness();
           println!("Goodness: {g1}");
           if g1==g0 {
               break
           }
           g0=g1
        }

        let x = net.get_state();
        let mut mind=LLEN+1;
        let mut mini=0;
        for (i,v) in cb.iter().enumerate() {
            let d: usize=v.iter().zip(&x[0..v.len()]).map(|(x,y)| if x==y {0} else {1}).sum();
            println!("lab: {lab} cb: {i} d: {d}");
            if d<mind {
                mind=d;
                mini=i;
            }
        }
        n+=1;
        if *lab as usize ==mini {
            correct+=1;
        }
        println!("correct {correct}/{n}");
    }
}

fn hop_font8x8() {
    let mut net = HopfieldNet::new(&[0; 64]);
    if true {
        for i in 0x61..0x64 {
            let b = font8x8::unicode2bitmap(i);
            font8x8::display(b);
            let v = u64_to_state(b);
            net.set_state(&v);
            net.hopfield_storage_rule();
        }
    } else {
        for _ in 0..3 {
            for i in 0x61..0x64 {
                let b = font8x8::unicode2bitmap(i);
                font8x8::display(b);
                let v = u64_to_state(b);
                net.set_state(&v);
                for _ in 0..10 {
                    let change = net.perceptron_conv_procedure();
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
            let v = u64_to_state(b);
            net.set_state(&v);
            println!("Goodness: {}", net.goodness());
            for _ in 0..3 {
                net.step();
                println!("Goodness: {}", net.goodness());
            }
            let v = net.get_state();
            let b = state2u64(v);
            font8x8::display(b);
        }
    }
}

fn main() {
    //hop_font8x8();
    mnist_classification();
}
