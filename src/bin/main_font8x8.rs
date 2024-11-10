use hopfield::hopfield::Hopfield;

fn state2u64(state: &[u8]) -> u64 {
    state
        .iter()
        //.skip(1) // bias term
        .enumerate()
        .fold(0, |b, (i, &x)| b | ((x as u64) << i))
}

fn u64_to_state(u: u64) -> [u8; 64] {
    let mut a: [u8; 64] = [0; 64];
    a.iter_mut()
        .enumerate()
        .for_each(|(i, e)| *e = if u & (1 << i) != 0 { 1 } else { 0 });
    a
}

fn hop_font8x8() {
    let mut net = Hopfield::<64>::new();
    if true {
        for i in 0x61..0x64 {
            let b = font8x8::unicode2bitmap(i);
            font8x8::display(b);
            let v = u64_to_state(b);
            net.hopfield_storage_rule(&v);
        }
    } else {
        for ep in 0..3 {
            for i in 0x61..0x64 {
                let b = font8x8::unicode2bitmap(i);
                if ep == 0 {
                    font8x8::display(b);
                }
                let v = u64_to_state(b);
                for _ in 0..10 {
                    net.perceptron_conv_procedure(&v);
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
    hop_font8x8();
}
