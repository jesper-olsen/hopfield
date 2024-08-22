use core::fmt;

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

struct HopfieldNet {
    weights: Vec<i32>,
    state: Vec<u8>,
}

impl fmt::Display for HopfieldNet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "HopfieldNet (State 0 is bias)")?;
        for i in 0..self.state.len() {
            write!(f, "State {i:>2}({:>2}): ", self.state[i])?;
            for j in 0..self.state.len() {
                write!(f, "{:>3} ", self.get_weight(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl HopfieldNet {
    fn new(state: &[u8]) -> Self {
        // add a bias term - state that is always on
        let size = state.len() + 1;
        let mut s0 = Vec::with_capacity(size);
        s0.push(1);
        s0.extend_from_slice(state);
        let num_elements = (size * (size - 1)) / 2;
        Self {
            weights: vec![0; num_elements],
            state: s0,
        }
    }

    fn index(&self, i: usize, j: usize) -> usize {
        //assert!(i < j);
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        let size = self.state.len();
        (i * (2 * size - i - 1)) / 2 + (j - i - 1)
    }

    fn set_weight(&mut self, i: usize, j: usize, value: i32) {
        if i != j {
            let index = self.index(i, j);
            self.weights[index] = value;
        }
    }

    fn get_weight(&self, i: usize, j: usize) -> i32 {
        if i == j {
            0
        } else {
            let index = self.index(i, j);
            self.weights[index]
        }
    }

    fn update_weight(&mut self, i: usize, j: usize, delta: i32) {
        if i != j {
            let index = self.index(i, j);
            self.weights[index] += delta;
        }
    }

    fn hopfield_storage_rule(&mut self) {
        // Hopfield with -1 & 1 states
        //     delta w_ij = s_i * s_j
        // Hopfield with 0 & 1 states
        //     delta w_ij = 4(s_i-0.5)(s_j-0.5)
        //                = 4(s_i*s_j -0.5s_i-0.5*s_j+0.25)
        //                = 4s_i*s_j -2s_i -2s_j + 1
        // For M memories, weights in range [-M;M] 

        for i in 1..self.state.len() {
            let si: i32 = self.state[i].into();
            for j in 0..self.state.len() {
                let sj: i32 = self.state[j].into();
                let dw: i32 = 4 * si * sj - 2 * si - 2 * sj + 1;
                self.update_weight(i, j, dw)
            }
        }
    }

    fn perceptron_conv_procedure(&mut self) -> bool {
        //* if output unit is correct do nothing
        //* if incorrectly outputs zero, add input vector to weight vector
        //* if incorrectly outputs one, subtract input vector from weight vector 

        let size = self.state.len();
        let mut change = false;
        for i in 1..size {
            let e = self
                .state
                .iter()
                .enumerate()
                .map(|(j, &sj)| sj as i32 * self.get_weight(i, j))
                .sum::<i32>();
            let sign = match self.state[i] {
                0 if e >= 0 => -1,
                1 if e < 0 => 1,
                _ => 0,
            };
            if sign != 0 {
                for j in 0..size {
                    let w: i32 = self.state[j].into();
                    self.update_weight(i, j, sign * w);
                }
                change = true;
            }
        }
        change
    }

    fn step(&mut self) {
        let size = self.state.len();
        for i in 1..size {
            let e = self
                .state
                .iter()
                .enumerate()
                .map(|(j, &sj)| i32::from(sj) * self.get_weight(i, j))
                .sum::<i32>();
            self.state[i] = if e < 0 { 0 } else { 1 };
        }
    }

    fn goodness(&self) -> i32 {
        -self.energy()
    }

    fn energy(&self) -> i32 {
        -(0..self.state.len())
            .flat_map(|i| {
                (0..i).map(move |j| {
                    self.state[i] as i32 * self.state[j] as i32 * self.get_weight(i, j)
                })
            })
            .sum::<i32>()
    }

    fn set_state(&mut self, state: &[u8]) {
        assert_eq!(self.state.len(), state.len() + 1);
        self.state[1..].copy_from_slice(state);
    }

    fn get_state(&self) -> &[u8] {
        &self.state[1..]
    }
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
    for (label,mask) in [("none",mask0), ("upper",mask1),("lower",mask2)] {
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

#[cfg(test)]
mod tests {
    use crate::HopfieldNet;

    #[test]
    fn h_test() {
        let mut net = HopfieldNet::new(&[0; 5]);
        net.set_weight(1, 2, -4);
        net.set_weight(1, 4, 3);
        net.set_weight(1, 5, 3);
        net.set_weight(2, 3, 3);
        net.set_weight(2, 4, 2);
        net.set_weight(3, 4, -1);
        net.set_weight(4, 5, -1);

        net.set_state(&[0, 1, 1, 0, 0]);
        assert_eq!(net.goodness(), 3);
        net.step();
        assert_eq!(net.goodness(), 4);

        net.set_state(&[0, 1, 1, 1, 0]);
        assert_eq!(net.goodness(), 4);

        net.set_state(&[1, 0, 0, 1, 1]);
        assert_eq!(net.goodness(), 5);
    }
}
