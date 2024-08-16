use core::fmt;
//use core::ops::Mul;
//use std::default::Default;
//use std::ops::{AddAssign, Sub};

fn state2u64(state: &[u8]) -> u64 {
    let mut b: u64 = 0;
    for (i,x) in state.iter().enumerate() {
        if *x !=0 {
            b|=1<<i
        } 
    } 
    b
}

struct HopfieldNet {
    weights: Vec<i32>,
    state: Vec<u8>,
}


impl fmt::Display for HopfieldNet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f,"HopfieldNet (State 0 is bias)")?;
        for i in 0..self.state.len() {
            write!(f, "State {i:>2}({:>2}): ", self.state[i])?;
            for j in 0..self.state.len() {
                write!(f, "{:>3} ", self.get_weight(i,j))?;
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
        let mut s0 = vec![1; size];
        for i in 1..size {
            s0[i] = state[i - 1]
        }
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
            let index = self.index(i,j);
            self.weights[index] = value;
        }
    }

    fn get_weight(&self, i: usize, j: usize) -> i32 {
        if i == j {
            0
        } else {
            let index = self.index(i,j);
            self.weights[index]
        }
    }

    fn update_weight(&mut self, i: usize, j: usize, delta: i32) {
        if i != j {
            let index = self.index(i,j);
            self.weights[index] += delta;
        }
    }

    fn storage_rule(&mut self, s: u64) {
        // 4*(si-0.5)(sj-0.5)
        for i in 0..64 {
            let si: i8 = ((s >> i) & 1).try_into().unwrap();
            for j in 0..64 {
                let sj: i8 = ((s >> j) & 1).try_into().unwrap();
                let dw: i32 = (4 * si * sj - 2 * si - 2 * sj + 1).into();
                self.update_weight(i, j, dw)
            }
        }
    }

    fn step(&mut self) {
        let size = self.state.len();
        for i in 1..size {
            let mut e = 0;
            for j in 0..size {
                let sj: i32 = self.state[j].try_into().unwrap();
                e += sj * self.get_weight(i, j);
            }
            self.state[i] = if e < 0 { 0 } else { 1 };
        }
    }

    fn goodness(&self) -> i32 {
        -self.energy()
    }

    fn energy(&self) -> i32 {
        let mut e: i32 = 0;
        for i in 0..self.state.len() {
            for j in 0..i {
                e -= self.state[i] as i32 * self.state[j] as i32 * self.get_weight(i, j);
            }
        }
        e
    }

    fn set_state(&mut self, state: &[u8]) {
        assert_eq!(self.state.len(), state.len() + 1);
        for i in 1..self.state.len() {
            self.state[i] = state[i - 1]
        }
    }

    fn get_state(&self) -> &[u8] {
       &self.state[1..]
    }
}

fn main() {
    let mut net = HopfieldNet::new(&[0; 5]);
    net.set_weight(1, 2, -4);
    net.set_weight(1, 4, 3);
    net.set_weight(1, 5, 3);
    net.set_weight(2, 3, 3);
    net.set_weight(2, 4, 2);
    net.set_weight(3, 4, -1);
    net.set_weight(4, 5, -1);
    net.set_state(&[0, 1, 1, 0, 0]); // goodness 3
    println!("{net}");
    println!("bm {}", state2u64(net.get_state()));
    println!("goodness {}", net.goodness());
    net.set_state(&[0, 1, 1, 1, 0]); // goodness 4
    println!("goodness {}", net.goodness());
    net.set_state(&[1, 0, 0, 1, 1]); // goodness 5
    println!("goodness {}", net.goodness());
    net.set_state(&[0, 1, 1, 0, 0]); // goodness 3
    net.step();
    println!("goodness {}", net.goodness());
    net.step();
    println!("goodness {}", net.goodness());
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
