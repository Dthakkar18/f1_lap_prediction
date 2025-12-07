# Machine Learning Model for Lap Time Prediction

## Goal:
Build an ML model to predict lap times based on fuel load, tire wear, track temperature, air temperature and weather conditions.

## Process:
### Load dataset
### Engineer additional features
### Clean dataset
### Define X (features) and y (target)
### Separate categorical and numeric columns
### Build preprocessing (OneHotEncoder + passthrough)
### Split X and y into train/test sets
### Create a Pipeline with preprocessing + algorithm
### Train the model on training data
### Predict using X_test
### Evaluate predictions against y_test

## Dependencies:
- python 
- pandas 
- numpy
- scikit-learn 
- fastf1
- matplotlib
- seaborn

## Next step:
- save results into csv
- make this functionally cleaner
- gather larger datasets (ie, entire 2023 season lap times)
- potentially transition to use XGBoost (better predictor with larger datasets)