use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

fn convert_to_256_color_scale(value: &f64) -> u8 {
    (value * 255.0) as u8
}

pub fn plot_png(
    name: &str,
    dims: (usize, usize),
    data: &Vec<Vec<f64>>,
    color_type: png::ColorType,
) -> Result<(), std::io::Error> {
    let path = Path::new("images").join(name).with_extension("png");

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = File::create(&path)?;
    let ref mut w = BufWriter::new(file);

    let width = dims.0 as u32;
    let height = dims.1 as u32;

    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(color_type);

    let mut writer = encoder.write_header()?;

    let data_uint8 = data
        .iter()
        .flatten()
        .map(convert_to_256_color_scale)
        .collect::<Vec<u8>>();

    writer.write_image_data(&data_uint8)?;

    Ok(())
}
