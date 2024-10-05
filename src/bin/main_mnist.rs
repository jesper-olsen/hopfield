use hopfield::{cnn, mnist, HopfieldNet};

const NUM_LABELS: usize = 10; // Number of labels
const Q: u8 = 2; // Quantization levels, e.g. 2, 4, 8
const D: u8 = (256usize / Q as usize) as u8;
const SS: usize = Q as usize * 28 * 28 + NUM_LABELS; // state length
const DIR: &str = "MNIST/";

fn image_to_state(label: &[u8], im: &[u8]) -> Vec<u8> {
    // one-hot encode intensity
    let p: Vec<u8> = im
        .iter()
        .flat_map(|&byte| {
            let bin = byte / D; // D bins
            (0..Q).map(move |i| if i == bin { 1 } else { 0 })
        })
        .collect();

    let mut x = Vec::with_capacity(label.len() + p.len());
    x.extend_from_slice(label);
    x.extend_from_slice(&p);
    x
}

fn state_to_image(state: &[u8], label_len: usize) -> Vec<u8> {
    let state = &state[label_len..]; // Skip label

    state
        .chunks(Q as usize) // Each pixel is represented by D one-hot values
        .map(|bin| {
            // Decode the one-hot encoding back into intensity
            bin.iter()
                .enumerate()
                .find(|(_, &v)| v == 1) // Find which bit is set to 1
                .map(|(i, _v)| (i as u8) * D) // Map back to intensity (0 to 255, D bins)
                .unwrap_or(0)
        })
        .collect()
}

//const fn one_hot<const N: usize>(i: usize) -> [u8; N] {
//    let mut a: [u8; N] = [0; N];
//    a[i] = 1;
//    a
//}

fn generate_one_hot_state_vectors<const NUM_LABELS: usize>() -> [[u8; NUM_LABELS]; NUM_LABELS] {
    let mut a = [[0; NUM_LABELS]; NUM_LABELS];
    for i in 0..NUM_LABELS {
        a[i][i] = 1;
    }
    a
}

fn mnist_train(nepochs: usize) {
    let cb = generate_one_hot_state_vectors::<NUM_LABELS>();
    //for (i, v) in cb.iter().enumerate() {
    //    println!("{i}: {v:?}")
    //}

    let fname = format!("{DIR}train-labels.idx1-ubyte");
    let labels = mnist::read_labels(&fname).unwrap();
    println!("Read {} labels", labels.len());

    let fname = format!("{DIR}train-images.idx3-ubyte");
    let images = mnist::read_images(&fname).unwrap();
    let mut net = HopfieldNet::<SS>::new();
    for j in 0..nepochs {
        for (i, (im, lab)) in images.iter().zip(labels.iter()).enumerate() {
            //mnist::plot_image(im, 28,28,*lab);
            let x = image_to_state(&cb[*lab as usize], im.as_u8_array());
            net.perceptron_conv_procedure(&x);
            if i % 100 == 0 {
                println!("{j},{i}");
            }
        }
        let fname = format!("hop{j}.txt");
        net.save_json(&fname).expect("failed to save");
    }
    mnist_test(&cb, &net)
}

fn predict(cb: &[[u8; NUM_LABELS]], x: &[u8]) -> usize {
    let mut mind = NUM_LABELS + 1;
    let mut mini = 0;
    for (i, v) in cb.iter().enumerate() {
        let d: usize = v
            .iter()
            .zip(&x[0..v.len()])
            .map(|(x, y)| if x == y { 0 } else { 1 })
            .sum();
        if d < mind {
            mind = d;
            mini = i;
        }
    }
    mini
}

// start with blank label and let the network reconstruct it as it settles in to an energy minimum
fn classify(net: &HopfieldNet<SS>, cb: &[[u8; NUM_LABELS]], x: &mut [u8], lab: u8) -> usize {
    let mut g0 = net.goodness(x);
    println!("Goodness: {g0}");
    let mut mini;
    loop {
        net.step(x);
        let g1 = net.goodness(x);
        mini = predict(cb, x);
        println!("Goodness: {g1}, lab: {lab} prediction: {mini}");
        if g1 == g0 {
            break;
        }
        g0 = g1
    }
    mini
}

fn mnist_test(cb: &[[u8; NUM_LABELS]], net: &HopfieldNet<SS>) {
    let fname = format!("{DIR}t10k-labels.idx1-ubyte");
    let labels = mnist::read_labels(&fname).unwrap();
    println!("Read {} labels", labels.len());

    let fname = format!("{DIR}t10k-images.idx3-ubyte");
    let images = mnist::read_images(&fname).unwrap();

    let mut correct = 0;
    let mut n = 0;
    for (im, lab) in images.iter().zip(labels.iter()) {
        //mnist::plot_image(im, 28,28,*lab);
        let mut x = image_to_state(&[0; 10], im.as_u8_array());

        let predicted_label = classify(net, cb, &mut x, *lab);
        n += 1;
        if *lab as usize == predicted_label {
            correct += 1;
        }
        println!("correct {correct}/{n}");

        //let imp=state_to_image(&x, 10);
        //println!("im len: {}, imp len: {} x len {}", im.len(), imp.len(), x.len());
        //mnist::plot_image(&im, 28,28,*lab);
        //mnist::plot_image(&imp, 28,28,*lab);
    }
}

fn main() {
    //mnist::test().expect("Failed");
    mnist_train(1);

    //let fname = format!("hop0.json");
    //let mut net = HopfieldNet::<SS>::load_json(&fname).expect("Failed to load Hopfield network");
    //let cb = generate_one_hot_state_vectors::<NUM_LABELS>();
    //mnist_test(&cb, &net);
}
