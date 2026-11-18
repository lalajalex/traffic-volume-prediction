# Traffic Volume Prediction

Predicting hourly westbound I-94 traffic volume (Minneapolis–St. Paul) from time, weather, and holiday data using regression models.

## Status
🚧 In progress (EDA)

- [x] Load and inspect data
- [x] Clean temperature outliers
- [x] Remove duplicate timestamps
- [ ] Explore traffic patterns
- [ ] Baseline and regression models
- [ ] Results and evaluation

## Dataset

[Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) — 48,204 hourly records (2012–2018) of traffic volume with weather and holiday features.

John Hogue, 2019, *Metro Interstate Traffic Volume* [Dataset], UCI Machine Learning Repository, DOI 10.24432/C5X60B. Licensed under CC BY 4.0.

## Data Cleaning so far
- Replaced impossible 'temp' values of 0K with linear interpolation after sorting by time
- Removed duplicate timestamps
- Found a 10 month gap in the data (Aug 2014 - June 2015 = 307 days)

## Project Structure

data/ raw dataset
notebooks/ EDA and modeling notebooks
src/ reusable code
figures/ saved charts

## Team
- Alexander Lalaj
- Domenic Nikprelevic