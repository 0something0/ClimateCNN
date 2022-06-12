# ClimateCNN 
A GIS processing and visualization tool

`poolsville_hacks` 2022 Best Overall winner! 🎉🎉🎉

See how a world of your own making, or Earth in the future, or your favorite fictional world, would look like from space! 

ClimateCNN uses a neural network to predict how a certain area on an Earthlike planet might look like given the area's topographic, temperature, and precipitation data. 

## Installation

Download this repository or run
```bash
git clone https://github.com/0something0/ClimateCNN
```
### Dependencies

Currently, ClimateCNN requires [Conda](https://docs.conda.io/en/latest/) for its [Rasterio](https://rasterio.readthedocs.io/en/latest/) and [GDAL](https://rasterio.readthedocs.io/en/latest/) dependencies. 

After, run
```bash
pip install -r requirements.txt
```
in the project directory to download the dependencies.

## Usage

ClimateCNN uses GeoTIFF files for rainfall, temperature, and heightmap data, and the PNG files for colormaps. 

Currently, the project only supports inputs of one file per folder.


### Training

This is not needed if you are merely using the provided neural network model, rather than training your own. 

If you are training your own model, you should provide every folder with the corrosponding data.

```
├── training_dataset
│   ├── color
│   ├── elevation
│   ├── rainfall
│   └── temperature
```


### Prediction 

Only the elevation, rainfall, and temperature folders should be given input files. Color_output will output prediction data in chunks.

```
.
├── prediction_dataset
│   ├── color_output
│   ├── elevation
│   ├── rainfall
│   └── temperature

```
Single, contigious files should be provided for all inputs.

## Roadmap

- [ ] Remove GDAL dependencies
- [ ] Multiple file input support
- [ ] Experiment with different parameters
- [ ] Interactivity

## Works Cited


Amante, C. ETOPO1 1 Arc-Minute Global Relief Model: Procedures, Data Sources and Analysis. National Geophysical Data Center, NOAA, 2009. DOI.org (Datacite), https://doi.org/10.7289/V5C8276M.

Fick, S.E. and R.J. Hijmans, 2017. WorldClim 2: new 1km spatial resolution climate surfaces for global land areas. International Journal of Climatology 37 (12): 4302-4315.

NASA Visible Earth - Home. https://visibleearth.nasa.gov/collection/1484/blue-marble.



