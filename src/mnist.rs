use gnuplot::{AxesCommon, Figure, Fix};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read};

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

pub fn read_images(path: &str) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let magic_number = read_u32(&mut reader)?;
    if magic_number != 2051 {
        return Err("Invalid magic number for image file".into());
    }

    let num_images = read_u32(&mut reader)?;
    let num_rows = read_u32(&mut reader)?;
    let num_cols = read_u32(&mut reader)?;
    let image_size = (num_rows * num_cols) as usize;

    let mut images = Vec::with_capacity(num_images as usize);
    for _ in 0..num_images {
        let mut image = vec![0u8; image_size];
        reader.read_exact(&mut image)?;
        images.push(image);
    }

    Ok(images)
}

fn plot(image: &[[u8; 28]; 28], label: u8) {
    let cols = image.len();
    let rows = image[0].len();
    let z: Vec<_> = image
        .iter()
        .rev()
        .flat_map(|r| r.iter())
        .map(|&p| p as f64)
        .collect();

    // Plot the image using a heatmap
    let mut fg = Figure::new();
    fg.axes2d()
        .set_aspect_ratio(Fix(1.0))
        .set_size(1.0, 1.0)
        .set_x_range(Fix(0.0), Fix(cols as f64))
        .set_y_range(Fix(0.0), Fix(rows as f64))
        .image(
            z.iter(),
            rows,
            cols,
            Some((0.0, 0.0, cols as f64, rows as f64)),
            &[],
        )
        .set_title(&format!("MNIST Label: {}", label), &[]); // Add the label as the title

    fg.show().unwrap();
}

pub fn plot_image(image: &[u8], rows: usize, cols: usize, label: u8) {
    let mut fg = Figure::new();

    let z: Vec<f64> = image
        .chunks(cols) // Split into rows
        .rev()
        .flat_map(|r| r.iter())
        .map(|&p| p as f64)
        .collect();

    // Plot the image using a heatmap
    fg.axes2d()
        .set_aspect_ratio(Fix(1.0))
        .set_size(1.0, 1.0)
        .set_x_range(Fix(0.0), Fix(cols as f64))
        .set_y_range(Fix(0.0), Fix(rows as f64))
        .image(
            z.iter(),
            rows,
            cols,
            Some((0.0, 0.0, cols as f64, rows as f64)),
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
        println!(
            "Read {} {s} images, each with {} pixels",
            images.len(),
            images[0].len()
        );
        for i in 0..5 {
            plot_image(&images[i], 28, 28, labels[i]);
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

pub fn test_plot() {
    let fname = format!("MNIST/train-labels.idx1-ubyte");
    let labels = read_labels(&fname).expect("Failed to read labels");

    let fname = format!("MNIST/train-images.idx3-ubyte");
    let images = read_images(&fname).expect("Failed to read images");
    //plot_image(&images[0], 28, 28, labels[0]);
    let image = unflatten_image(&images[0]);
    plot(&image, labels[0]);
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
            assert_eq!(images[0].len(), 784);
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
