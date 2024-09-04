use core::fmt;
pub mod mnist;
pub mod hopfield;
use hopfield::HopfieldNet;

fn state2u64(state: &[u8]) -> u64 {
    state
        .iter()
        .enumerate()
        .fold(0, |b, (i, &x)| b | ((x as u64) << i))
}

fn u64_to_state(b: u64) -> Vec<u8> {
    (0..64)
        .map(|i| if b & (1 << i) != 0 { 1 } else { 0 })
        .collect()
}

fn main() {
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
                    if !change {
                        break;
                    }
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
