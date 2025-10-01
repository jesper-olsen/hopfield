use hopfield::hopfield::Hopfield;
use hopfield::state::State;
use std::path::PathBuf;
use std::io::{self,Write};

const NUM_LABELS: usize = 10; // Number of labels
const Q: u8 = 2; // Quantization levels, e.g. 2, 4, 8
const D: u8 = (256usize / Q as usize) as u8;
const IDIM: usize = Q as usize * 28 * 28 + NUM_LABELS; // state length

fn image_to_state(label: &[u8], im: &[u8]) -> State<IDIM> {
    // one-hot encode intensity
    let p: Vec<u8> = im
        .iter()
        .flat_map(|&byte| {
            let bin = byte / D; // D bins
            (0..Q).map(move |i| if i == bin { 1 } else { 0 })
        })
        .collect();

    let mut x: [u8; IDIM] = [0; IDIM];
    x[0..label.len()].copy_from_slice(label);
    x[label.len()..].copy_from_slice(&p);
    State::from_bool_slice(&x)
}

fn generate_one_hot_state_vectors<const NUM_LABELS: usize>() -> [[u8; NUM_LABELS]; NUM_LABELS] {
    std::array::from_fn(|i| {
        let mut row = [0; NUM_LABELS];
        row[i] = 1;
        row
    })
}

fn mnist_train(nepochs: usize) {
    let cb = generate_one_hot_state_vectors::<NUM_LABELS>();

    let dir: PathBuf = PathBuf::from("MNIST/");
    let fname = dir.join("train-labels-idx1-ubyte");
    let labels = mnist::read_labels(&fname).unwrap();
    println!("Read {} labels", labels.len());

    let fname = dir.join("train-images-idx3-ubyte");
    let images = mnist::read_images(&fname).unwrap();

    let mut net = Hopfield::<IDIM>::new();
    for j in 0..nepochs {
        for (i, (im, lab)) in images.iter().zip(labels.iter()).enumerate() {
            let x = image_to_state(&cb[*lab as usize], im.as_u8_array());
            net.perceptron_conv_procedure(&x);
            if i % 100 == 0 {
                print!("Epoch {j}, Image {i:5}    \r");
                let _ = io::stdout().flush();
            }
        }
        let fname = format!("hop{j}.json");
        net.save_json(&fname).expect("failed to save");
    }
    mnist_test(&cb, &net)
}

fn predict(cb: &[[u8; NUM_LABELS]], x: &State<IDIM>) -> usize {
    let mut mind = NUM_LABELS + 1;
    let mut mini = 0usize;
    for (i, v) in cb.iter().enumerate() {
        let mut d = 0usize;
        for j in 0..v.len() {
            if v[j]!=x.get(j) {
                d+=1;
            }
        }
        if d < mind {
            mind = d;
            mini = i;
        }
    }
    mini
}

// start with blank label and let the network reconstruct it as it settles in to an energy minimum
fn classify(net: &Hopfield<IDIM>, cb: &[[u8; NUM_LABELS]], x: &mut State<IDIM>, lab: u8) -> usize {
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

fn mnist_test(cb: &[[u8; NUM_LABELS]], net: &Hopfield<IDIM>) {
    let dir: PathBuf = PathBuf::from("MNIST/");
    let fname = dir.join("t10k-labels-idx1-ubyte");
    let labels = mnist::read_labels(&fname).unwrap();
    println!("Read {} labels", labels.len());

    let fname = dir.join("t10k-images-idx3-ubyte");
    let images = mnist::read_images(&fname).unwrap();

    let mut correct = 0;
    for (n, (im, lab)) in images.iter().zip(labels.iter()).enumerate() {
        //mnist::plot_image(im, 28,28,*lab);
        let mut x = image_to_state(&[0; 10], im.as_u8_array());

        let predicted_label = classify(net, cb, &mut x, *lab);
        if *lab as usize == predicted_label {
            correct += 1;
        } else {
            println!("{im}");
        }
        println!("correct {correct}/{total}", total = n + 1);

        //let imp=state_to_image(&x, 10);
        //println!("im len: {}, imp len: {} x len {}", im.len(), imp.len(), x.len());
        //mnist::plot_image(&im, 28,28,*lab);
        //mnist::plot_image(&imp, 28,28,*lab);
    }
}

fn main() {
    //mnist::test().expect("Failed");
    mnist_train(1);

    // let fname = format!("hop0.json");
    // let mut net = Hopfield::<IDIM>::load_json(&fname).expect("Failed to load Hopfield network");
    // let cb = generate_one_hot_state_vectors::<NUM_LABELS>();
    // mnist_test(&cb, &net);
}
