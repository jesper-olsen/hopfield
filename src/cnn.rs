use stmc_rs::mersenne::MersenneTwister64;

struct Kernel<const W: usize> {
    w: [[i32; W]; W],
}

impl<const W: usize> Kernel<W> {
    fn new() -> Self {
        let mut w = [[0i32; W]; W];
        for i in 0..W {
            w[i][0] = 1;
            w[i][W - 1] = -1;
        }
        Kernel { w }
    }

    //fn convolve<const IMAGE_WIDTH: usize>(&self, image: &[u8]) -> Vec<i32> {
    //    let osize: usize = IMAGE_WIDTH - W + 1;
    //    let mut output = vec![0; osize * osize];

    //    for y in 0..osize {
    //        for x in 0..osize {
    //            let mut sum = 0;
    //            for ky in 0..W {
    //                for kx in 0..W {
    //                    let image_y = y * stride + ky;
    //                    let image_x = x * stride + kx;
    //                    if image_y >= pad && image_y < padded_width - pad
    //                        && image_x >= pad && image_x < padded_width - pad
    //                    {
    //                        let image_index = (image_y - pad) * IMAGE_WIDTH + (image_x - pad);
    //                        sum += image[image_index] as i32 * self.w[ky][kx];
    //                    }

    //                    let image_index = (y + ky) * IMAGE_WIDTH + (x + kx);
    //                    sum += image[image_index] as i32 * self.w[ky][kx];
    //                }
    //            }
    //            output[y * osize + x] = sum;
    //        }
    //    }
    //    output
    //}

    fn convolve<const IMAGE_WIDTH: usize, const STRIDE: usize, const PAD: usize>(
        &self,
        image: &[u8],
    ) -> Vec<i32> {
        let padded_width = IMAGE_WIDTH + 2 * PAD;
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
                            let image_index = (image_y - PAD) * IMAGE_WIDTH + (image_x - PAD);
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
    //fn update<const IMAGE_WIDTH: usize>(&mut self, image: &[u8], lr: f32) {
    //    let conv_output = self.convolve::<IMAGE_WIDTH>(image);
    //    for y in 0..(IMAGE_WIDTH - W + 1) {
    //        for x in 0..(IMAGE_WIDTH - W + 1) {
    //            let conv_value = conv_output[y * (IMAGE_WIDTH - W + 1) + x];
    //            for ky in 0..W {
    //                for kx in 0..W {
    //                    let image_index = (y + ky) * IMAGE_WIDTH + (x + kx);
    //                    self.w[ky][kx] +=
    //                        (lr * conv_value as f32 * image[image_index] as f32) as i32;
    //                }
    //            }
    //        }
    //    }
    //}

    /// Hebbian update
    fn update<const IMAGE_WIDTH: usize, const STRIDE: usize, const PAD: usize>(
        &mut self,
        image: &[u8],
        lr: f32,
    ) {
        let conv_output = self.convolve::<IMAGE_WIDTH,STRIDE,PAD>(image);
        let osize = (IMAGE_WIDTH + 2 * PAD - W) / STRIDE + 1;

        for y in 0..osize {
            for x in 0..osize {
                let conv_value = conv_output[y * osize + x];
                for ky in 0..W {
                    for kx in 0..W {
                        let image_y = y * STRIDE + ky;
                        let image_x = x * STRIDE + kx;

                        if image_y < IMAGE_WIDTH && image_x < IMAGE_WIDTH {
                            let image_index = image_y * IMAGE_WIDTH + image_x;
                            self.w[ky][kx] +=
                                (lr * conv_value as f32 * image[image_index] as f32) as i32;
                        }
                    }
                }
            }
        }
    }

    /// Train Hebbian
    fn train<const IMAGE_WIDTH: usize>(&mut self, images: &[Vec<u8>], lr: f32) {
        const STRIDE: usize = 1;
        const PAD: usize = 0;
        for (i, image) in images.iter().enumerate() {
            self.update::<IMAGE_WIDTH,STRIDE,PAD>(image, lr);
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
    const KERNEL_WIDTH: usize = 3;
    const STRIDE: usize = 1;
    const PAD: usize = 0;

    let image: [u8; IMAGE_WIDTH * IMAGE_WIDTH] = std::array::from_fn(|i| (i + 1) as u8);
    //let image: [u8; IMAGE_WIDTH * IMAGE_WIDTH] = [
    //    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    //];

    let kernel: [i32; KERNEL_WIDTH * KERNEL_WIDTH] = [1, 0, -1, 1, 0, -1, 1, 0, -1];

    let kernel = Kernel::<3>::new();
    let result = kernel.convolve::<IMAGE_WIDTH,STRIDE,PAD>(&image);
    result
    //println!("{:?}", result);

    //let mut mt =
    //    //MersenneTwister64::new(43);
    //    MersenneTwister64::new_init_by_array(&[0x12345, 0x23456, 0x34567, 0x45678]);
    //for x in mt.genrand_array::<{KERNEL_WIDTH*KERNEL_WIDTH}>().iter() {
    //    println!("{x}");
    //}
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
