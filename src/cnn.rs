use core::fmt;
use stmc_rs::mersenne::MersenneTwister64;

pub struct Kernel<const W: usize> {
    w: [[i32; W]; W],
}

impl<const W: usize> fmt::Display for Kernel<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CNN {W}x{W} Kernel ")?;

        for i in 0..W {
            for j in 0..W {
                write!(f, "{:>3} ", self.w[i][j])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl<const W: usize> Kernel<W> {
    // random elements drawn from -1, 0, 1
    pub fn new(seed: u64) -> Self {
        let mut mt = MersenneTwister64::new(seed);
        let mut w = [[0i32; W]; W];
        for i in 0..W {
            for j in 0..W {
                w[i][j] = (mt.genrand() % 2) as i32 - 1;
            }
        }
        Kernel { w }
    }

    /// edges 1-4
    pub fn edge(n: u8) -> Self {
        let mut w = [[0i32; W]; W];
        for i in 0..W {
            match n {
                // vertical edge
                1 => {
                    w[i][0] = 1;
                    w[i][W - 1] = -1;
                }
                // horizontal edge
                2 => {
                    w[0][i] = 1;
                    w[W - 1][i] = -1;
                }
                // diagonal - UL-LR
                3 => w[i][i] = 1,
                // diagonal - LL-UR
                _ => w[W - i - 1][i] = 1,
            }
        }
        Kernel { w }
    }

    fn convolve<const STRIDE: usize, const PAD: usize>(&self, image: &[u8], image_width: usize) -> Vec<i32> {
        let padded_width = image_width + 2 * PAD;
        let osize: usize = (padded_width - W) / STRIDE + 1;
        let mut output = vec![0; osize * osize];

        for y in 0..osize {
            for x in 0..osize {
                let mut sum = 0;
                for ky in 0..W {
                    for kx in 0..W {
                        // Determine the coordinates of the image with padding applied
                        let image_y = y * STRIDE + ky;
                        let image_x = x * STRIDE + kx;

                        // Only convolve if within the padded image bounds
                        if image_y >= PAD
                            && image_y < padded_width - PAD
                            && image_x >= PAD
                            && image_x < padded_width - PAD
                        {
                            let image_index = (image_y - PAD) * image_width + (image_x - PAD);
                            sum += image[image_index] as i32 * self.w[ky][kx];
                        }
                    }
                }
                output[y * osize + x] = sum;
            }
        }
        output
    }

    /// Hebbian update
    fn update<const STRIDE: usize, const PAD: usize>(&mut self, image: &[u8], image_width: usize, lr: f32) {
        let conv_output = self.convolve::<STRIDE, PAD>(image, image_width);
        let osize = (image_width + 2 * PAD - W) / STRIDE + 1;

        for y in 0..osize {
            for x in 0..osize {
                let conv_value = conv_output[y * osize + x];
                for ky in 0..W {
                    for kx in 0..W {
                        let image_y = y * STRIDE + ky;
                        let image_x = x * STRIDE + kx;

                        if image_y < image_width && image_x < image_width {
                            let image_index = image_y * image_width + image_x;
                            self.w[ky][kx] +=
                                (lr * conv_value as f32 * image[image_index] as f32) as i32;
                        }
                    }
                }
            }
        }
    }

    /// Train Hebbian
    fn train(&mut self, images: &[Vec<u8>], image_width: usize, lr: f32) {
        const STRIDE: usize = 1;
        const PAD: usize = 0;
        for (_i, image) in images.iter().enumerate() {
            self.update::<STRIDE, PAD>(image, image_width, lr);
        }
        self.normalise(); // Optional - prevent exploding weights
    }

    fn normalise(&mut self) {
        let norm = (self.w.iter().flatten().map(|&w| w * w).sum::<i32>() as f64).sqrt();
        if norm > 0.0 {
            for e in self.w.iter_mut().flatten() {
                *e = ((*e as f64) / norm) as i32;
            }
        }
    }
}

pub fn test_convolve() -> Vec<i32> {
    const IMAGE_WIDTH: usize = 5;
    const IMAGE_HEIGHT: usize = 5;
    const STRIDE: usize = 1;
    const PAD: usize = 0;

    let image: [u8; IMAGE_WIDTH * IMAGE_WIDTH] = std::array::from_fn(|i| (i + 1) as u8);
    let kernel = Kernel::<3>::edge(1);
    let result = kernel.convolve::<STRIDE, PAD>(&image, IMAGE_WIDTH);
    result
    //println!("{:?}", result);
}

#[cfg(test)]
mod tests {
    use crate::cnn::test_convolve;

    #[test]
    fn k_test() {
        let v = test_convolve();
        assert_eq!(v, vec![-6; 9]);
    }
}
