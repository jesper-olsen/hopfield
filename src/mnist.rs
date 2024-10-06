use gnuplot::{AxesCommon, Figure, Fix};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read};

const IMAGE_WIDTH: usize = 28;
const IMAGE_HEIGHT: usize = 28;

pub struct Image {
    pixels: [u8; IMAGE_WIDTH * IMAGE_HEIGHT],
}

impl Image {
    pub fn as_u8_array(&self) -> &[u8] {
        &self.pixels
    }

    pub fn as_2d_array(&self) -> &[[u8; IMAGE_WIDTH]; IMAGE_HEIGHT] {
        unsafe { &*(self.pixels.as_ptr() as *const [[u8; IMAGE_WIDTH]; IMAGE_HEIGHT]) }
    }

    pub fn as_f32_array(&self) -> [f32; IMAGE_WIDTH * IMAGE_HEIGHT] {
        let r: Vec<f32> = self.pixels.iter().map(|i| (*i as f32) / 255.0).collect();
        r.try_into().expect("failed cast")
    }
}

fn read_u32(reader: &mut BufReader<File>) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

pub fn read_labels(path: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let magic_number = read_u32(&mut reader)?;
    if magic_number != 2049 {
        return Err("Invalid magic number for label file".into());
    }

    let num_items = read_u32(&mut reader)?;
    let mut labels = vec![0u8; num_items as usize];
    reader.read_exact(&mut labels)?;

    Ok(labels)
}

pub fn read_images(path: &str) -> Result<Vec<Image>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let magic_number = read_u32(&mut reader)?;
    if magic_number != 2051 {
        return Err("Invalid magic number for image file".into());
    }

    let num_images = read_u32(&mut reader)?;
    let num_rows = read_u32(&mut reader)?;
    let num_cols = read_u32(&mut reader)?;
    assert_eq!(num_rows as usize, IMAGE_HEIGHT);
    assert_eq!(num_cols as usize, IMAGE_WIDTH);

    let mut images = Vec::with_capacity(num_images as usize);
    for _ in 0..num_images {
        let mut pixels = [0u8; IMAGE_HEIGHT * IMAGE_WIDTH];
        reader.read_exact(&mut pixels)?;
        images.push(Image { pixels });
    }

    Ok(images)
}

pub fn plot(image: &Image, label: u8) {
    let mut fg = Figure::new();

    let z: Vec<f64> = image.pixels
        .chunks(IMAGE_WIDTH) // Split into rows
        .rev()
        .flat_map(|r| r.iter())
        .map(|&p| p as f64)
        .collect();

    // Plot the image using a heatmap
    fg.axes2d()
        .set_aspect_ratio(Fix(1.0))
        .set_size(1.0, 1.0)
        .set_x_range(Fix(0.0), Fix(IMAGE_WIDTH as f64))
        .set_y_range(Fix(0.0), Fix(IMAGE_HEIGHT as f64))
        .image(
            z.iter(),
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            Some((0.0, 0.0, IMAGE_WIDTH as f64, IMAGE_HEIGHT as f64)),
            &[],
        )
        .set_title(&format!("MNIST Label: {}", label), &[]); // Add the label as the title

    fg.show().unwrap();
}

pub fn test() -> Result<(), Box<dyn Error>> {
    let dir = "MNIST/";
    for s in ["train", "t10k"] {
        let fname = format!("{dir}{s}-labels.idx1-ubyte");
        let labels = read_labels(&fname)?;
        println!("Read {} {s} labels", labels.len());

        let fname = format!("{dir}{s}-images.idx3-ubyte");
        let images = read_images(&fname)?;
        let pixels = images[0].as_u8_array();
        println!(
            "Read {} {s} images, each with {} pixels",
            images.len(),
            pixels.len()
        );
        for i in 0..5 {
            plot(&images[i], labels[i]);
        }
    }

    Ok(())
}

fn flatten_image(image: &[[u8; 28]; 28]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(image.as_ptr() as *const u8, 28 * 28) }
}

fn unflatten_image(image: &[u8]) -> &[[u8; 28]; 28] {
    assert_eq!(image.len(), 28 * 28);
    unsafe { &*(image.as_ptr() as *const [[u8; 28]; 28]) }
}

#[cfg(test)]
mod tests {
    use crate::mnist::{read_images, read_labels};
    #[test]
    fn m_test() {
        let dir = "MNIST/";
        for s in ["train", "t10k"] {
            let fname = format!("{dir}{s}-labels.idx1-ubyte");
            let labels = read_labels(&fname).expect("Failed to read labels");

            let fname = format!("{dir}{s}-images.idx3-ubyte");
            let images = read_images(&fname).expect("Failed to read images");
            assert_eq!(images[0].pixels.len(), 784);
            if s == "train" {
                assert_eq!(labels.len(), 60000);
                assert_eq!(images.len(), 60000);
            } else {
                assert_eq!(labels.len(), 10000);
                assert_eq!(images.len(), 10000);
            }
        }
    }
}
