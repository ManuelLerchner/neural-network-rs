use plotters::prelude::*;

pub fn plot_graph(name: String, data: &Vec<(i32, f64)>) -> Result<(), Box<dyn std::error::Error>> {
    let path_name = format!("images/{}_history.png", name);

    let root = BitMapBackend::new(&path_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_range = 0..data[data.len() - 1].0;
    let y_range = 0.0..data
        .iter()
        .map(|x| x.1)
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("History-".to_owned() + &name, ("sans-serif", 24).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(x_range, y_range)?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Cost")
        .draw()?;

    chart
        .draw_series(LineSeries::new((0..data.len()).map(|i| data[i]), &RED))?
        .label("cost")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
