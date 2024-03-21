# Building an application to predict real estate prices in Ninh Kiều - Cần Thơ

The dataset comprises 207 property records, with the price variable being the target, while the remaining 
columns contain property-related information such as usable area, width, number of bedrooms, number of bathrooms, and road frontage.

The primary objective is to train a regression machine learning model to generate more accurate cost estimates. 
As a regression problem, metrics such as the coefficient of determination (R-squared) and mean squared error will be used to evaluate the model's performance.
The predictions generated by this application assist users in assessing the potential value of properties they are 
interested in and making informed decisions regarding real estate investments in Ninh Kiều - Cần Thơ.

## Dataset visualization
![data](static/images/dulieu.png)
![data](static/images/tuongquan.png)

## Building regression models
### Regression Algorithms
- Criteria for Evaluating Regression Models: In practice, there are many criteria to evaluate regression models.
For example, evaluation based on absolute error value, relative error, based on squared error, etc.
- Here we will use:
  - MSE (Mean Squared Error): Represents the average squared difference between the actual value and the predicted value, extracted by squaring the difference.
  - MAE (Mean Absolute Error): Represents the average absolute error, indicating the difference between the original value and the predicted value, extracted by taking the average absolute difference in the dataset.
  - R-squared: Represents the fit of the model to the dataset. Similar to accuracy.
  - RMSE (Root Mean Squared Error): Calculated as the square root of MSE. Considered as the standard deviation of residuals (prediction errors).
Where:
  - 𝑦̂ is the predicted value of y.
  - 𝑦̅ is the mean value of y.

### Building Regression Models:
Firstly, the entire dataset will be split into Train and Test sets in an 80:20 ratio. 
With 8 parts used for model training and 2 parts used for testing.

```py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Results
After applying 6 regression algorithms including: Linear Regression, 
RandomForest Regression, DecisionTree Regressor, GradientBoosting Regressor, Neural Network Regression, Lasso Regression.

![data](static/images/ketqua.png)

Regression equation:

**muc_gia =  	-0.9789 + 0.0027*dt_san + 0.371*chieu_rong + 1.4762*phong_ngu + -0.5772*phong_tam + 0.1304*duong_vao**



