use core::fmt;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Read, Write};
pub mod mnist;

#[derive(Serialize, Deserialize)]
pub struct HopfieldNet<const SS: usize> {
    //weights: [i32; SS*(SS-1)/2],
    weights: Vec<i32>,
}

impl<const SS: usize> fmt::Display for HopfieldNet<SS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "HopfieldNet (State 0 is bias)")?;

        for i in 0..SS {
            for j in 0..SS {
                write!(f, "{:>3} ", self.get_weight(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl<const SS: usize> HopfieldNet<SS> {
    pub fn new() -> Self {
        Self {
            //weights: [0;SS*(SS-1)/2],
            weights: vec![0; SS * (SS - 1) / 2],
        }
    }

    pub fn save(&self, filename: &str) -> io::Result<()> {
        let encoded: Vec<u8> = bincode::serialize(self).map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Serialization error: {:?}", e),
            )
        })?;

        let mut file = File::create(filename)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    pub fn load_json(filename: &str) -> io::Result<HopfieldNet<SS>> {
        let mut file = File::open(filename)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        let hopfield_net: HopfieldNet<SS> = serde_json::from_str(&json)?;
        Ok(hopfield_net)
    }

    pub fn load(filename: &str) -> io::Result<HopfieldNet<SS>> {
        let mut file = File::open(filename)?;
        let mut encoded = Vec::new();
        file.read_to_end(&mut encoded)?;
        let hopfield_net: HopfieldNet<SS> =
            bincode::deserialize(&encoded).expect("Failed to deserialize HopfieldNet");
        Ok(hopfield_net)
    }

    pub fn save_json(&self, filename: &str) -> io::Result<()> {
        let json = serde_json::to_string(self)?;
        let mut file = File::create(filename)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    //pub fn save_weights(&self, filename: &str) -> io::Result<()> {
    //    let encoded: Vec<u8> = bincode::serialize(&self.weights)?;
    //    let mut file = File::create(filename)?;
    //    file.write_all(&encoded)?;
    //    Ok(())
    //}

    pub fn save_weights(&self, filename: &str) -> io::Result<()> {
        let encoded: Vec<u8> = bincode::serialize(&self.weights).map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Serialization error: {:?}", e),
            )
        })?;
        let mut file = File::create(filename)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    pub fn load_weights(&mut self, filename: &str) -> io::Result<()> {
        let mut file = File::open(filename)?;
        let mut encoded = Vec::new();
        file.read_to_end(&mut encoded)?;
        self.weights = bincode::deserialize(&encoded).expect("Failed to deserialize weights");
        Ok(())
    }

    fn index(&self, i: usize, j: usize) -> usize {
        //assert!(i < j);
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        (i * (2 * SS - i - 1)) / 2 + (j - i - 1)
    }

    pub fn set_weight(&mut self, i: usize, j: usize, value: i32) {
        if i != j {
            let index = self.index(i, j);
            self.weights[index] = value;
        }
    }

    pub fn get_weight(&self, i: usize, j: usize) -> i32 {
        if i == j {
            0
        } else {
            let index = self.index(i, j);
            self.weights[index]
        }
    }

    pub fn update_weight(&mut self, i: usize, j: usize, delta: i32) {
        if i != j {
            let index = self.index(i, j);
            self.weights[index] += delta;
        }
    }

    pub fn hopfield_storage_rule(&mut self, state: &[u8]) {
        // Hopfield with -1 & 1 states
        //     delta w_ij = s_i * s_j
        // Hopfield with 0 & 1 states
        //     delta w_ij = 4(s_i-0.5)(s_j-0.5)
        //                = 4(s_i*s_j -0.5s_i-0.5*s_j+0.25)
        //                = 4s_i*s_j -2s_i -2s_j + 1
        // For M memories, weights in range [-M;M]

        for i in 1..state.len() {
            let si: i32 = state[i].into();
            for j in 0..state.len() {
                let sj: i32 = state[j].into();
                let dw: i32 = 4 * si * sj - 2 * si - 2 * sj + 1;
                self.update_weight(i, j, dw)
            }
        }
    }

    pub fn perceptron_conv_procedure(&mut self, state: &[u8]) -> bool {
        //* if output unit is correct do nothing
        //* if incorrectly outputs zero, add input vector to weight vector
        //* if incorrectly outputs one, subtract input vector from weight vector

        let mut change = false;
        for i in 1..state.len() {
            let e = state
                .iter()
                .enumerate()
                .map(|(j, &sj)| sj as i32 * self.get_weight(i, j))
                .sum::<i32>();
            let sign = match state[i] {
                0 if e >= 0 => -1,
                1 if e < 0 => 1,
                _ => 0,
            };
            if sign != 0 {
                for j in 0..state.len() {
                    let w: i32 = state[j].into();
                    self.update_weight(i, j, sign * w);
                }
                change = true;
            }
        }
        change
    }

    pub fn step(&self, state: &mut [u8]) {
        for i in 1..state.len() {
            let e = state
                .iter()
                .enumerate()
                .map(|(j, &sj)| i32::from(sj) * self.get_weight(i, j))
                .sum::<i32>();
            state[i] = if e < 0 { 0 } else { 1 };
        }
    }

    pub fn goodness(&self, state: &[u8]) -> i32 {
        -self.energy(state)
    }

    pub fn energy(&self, state: &[u8]) -> i32 {
        -(0..state.len())
            .flat_map(|j| {
                (0..j).map(move |i| state[i] as i32 * state[j] as i32 * self.get_weight(i, j))
            })
            .sum::<i32>()
    }
}

#[cfg(test)]
mod tests {
    use crate::HopfieldNet;

    #[test]
    fn h_test() {
        let mut net = HopfieldNet::<6>::new();
        net.set_weight(1, 2, -4);
        net.set_weight(1, 4, 3);
        net.set_weight(1, 5, 3);
        net.set_weight(2, 3, 3);
        net.set_weight(2, 4, 2);
        net.set_weight(3, 4, -1);
        net.set_weight(4, 5, -1);

        let mut x = [1, 0, 1, 1, 0, 0];
        assert_eq!(net.goodness(&x), 3);
        net.step(&mut x);
        assert_eq!(net.goodness(&x), 4);

        let x = [1, 0, 1, 1, 1, 0];
        assert_eq!(net.goodness(&x), 4);

        let x = [1, 1, 0, 0, 1, 1];
        assert_eq!(net.goodness(&x), 5);
    }
}
