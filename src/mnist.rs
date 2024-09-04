use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, Read};

fn read_u32(reader: &mut BufReader<File>) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_labels(path: &str) -> Result<Vec<u8>, Box<dyn Error>> {
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

fn read_images(path: &str) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
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
    }

    Ok(())
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
