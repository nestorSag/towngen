# Generative models for city names by region

This project trains a transformer model on town name data and generate new, similar-sounding names. Most countries are available.

# Use

## Requirements

This project relies `keras-nlp` and Python 3.9

## Installation

Run `pip install -e .`

## Usage

### Download data

The `towngen.loaders` module downloads available data for a given country code. You can try the following example:

```py
import towngen.loaders.CityNames as CityNames

# load English town names
data = CityNames.load(
  country_code="GB", 
  query="admin_code1 == 'ENG'"
)
```

Downloaded data is cached in `.city_names`

### Train model

Use the `towngen` command to train the model and generate samples. use  `--help` to see available parameters.
