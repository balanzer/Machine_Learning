
# Linear regression

**Linear regression** is a basic predictive analytics technique that uses historical data to predict an output variable. It is popular for predictive modelling because it is easily understood and can be explained using plain English.

Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x).

Linear regression models have many real-world applications in an array of industries such as economics (e.g. predicting growth), business (e.g. predicting product sales, employee performance), social science (e.g. predicting political leanings from gender or race), healthcare (e.g. predicting blood pressure levels from weight, disease onset from biological factors), and more.

The basic idea is that if we can fit a linear regression model to observed data, we can then use the model to predict any future values. For example, let’s assume that we have found from historical data that the price (P) of a house is linearly dependent upon its size (S) — in fact, we found that a house’s price is exactly 90 times its size. The equation will look like this:

P = 90*S

With this model, we can then predict the cost of any house. If we have a house that is 1,500 square feet, we can calculate its price to be:
P = 90*1500 = $135,000

# Key Concepts 

### Overfitting: 
* The scenario when a machine learning model almost exactly matches the training data but performs very poorly when it encounters new data or validation set.

### Underfitting: 
* The scenario when a machine learning model is unable to capture the important patterns and insights from the data, which results in the model performing poorly on training data itself.

## Score or 𝑅²

* 𝑅² - It is also known as the coefficient of determination. This metric gives an indication of how good a model fits a given dataset.

* reg.score(X,y) - Return the coefficient of determination  of the prediction. 
* between 0 and 1, closer to 1 the better, eg. 0.9015975294607972 is 90%. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse)

* When you’re applying .score(), the arguments are also the predictor x and regressor y, and the return value is 𝑅².

* The low accuracy score from model suggests that our regressive model has not fitted very well to the existing data. High score means modal fit existing data well.

## intercept (intercept_)

* intercept_ and coefficient are used to predict new output values.

* The constant term in linear regression analysis seems to be such a simple thing. Also known as the y intercept, it is simply the value at which the fitted line crosses the y-axis.

* The slope(coef_) and intercept(intercept_) of the data are contained in the model's fit parameters.

* y=ax+b - where a is commonly known as the slope, and b is commonly known as the intercept.

## coefficient ( coef_)

* a numerical or constant quantity placed before and multiplying the variable in an algebraic expression (e.g. 4 in 4x y).

* each X value will have a coefficient 

* eg. like Price = 49.88  is intercept_, -0.113 for CRIM and 0.0611 for ZN are all its coefficient `Price = 49.88523466381753 + ( -0.11384484836914226 )*( CRIM )  + ( 0.061170268040606726 )*( ZN )  + ( 0.0541034649587423 )*( INDUS )  + ( 2.517511959122686 )*( CHAS )  + ( -22.248502345084425 )*( NX )  + ( 2.6984128200099033 )*( RM )  + ( 0.004836047284751289 )*( AGE )  + ( -1.5342953819992617 )*( DIS )  + ( 0.2988332548590185 )*( RAD )  + ( -0.011413580552025194 )*( TAX )  + ( -0.9889146257039411 )*( PTRATIO )  + ( -0.5861328508499092 )*( LSTAT )`


## Errors 

* Mean Absolute Error or MAE, Mean Squared Error or MSE or Root Mean Squared Error or RMSE

*  MAE is the sum of absolute differences between our target and predicted variables. error basically is the absolute difference between the actual or true values and the values that are predicted. Absolute difference means that if the result has a negative sign, it is ignored.

* Hence, MAE = True values – Predicted values

* MAE: It is not very sensitive to outliers in comparison to MSE since it doesn't punish huge errors. It is usually used when the performance is measured on continuous variable data. It gives a linear value, which averages the weighted individual differences equally. The lower the value, better is the model's performance.

* MSE is calculated by taking the average of the square of the difference between the original and predicted values of the data.

* MSE It is one of the most commonly used metrics, but least useful when a single bad prediction would ruin the entire model's predicting abilities, i.e when the dataset contains a lot of noise. It is most useful when the dataset contains outliers, or unexpected values (too high or too low values).

* RMSE is the standard deviation of the errors which occur when a prediction is made on a dataset. This is the same as MSE (Mean Squared Error) but the root of the value is considered while determining the accuracy of the model.

* RMSE, the errors are squared before they are averaged. This basically implies that RMSE assigns a higher weight to larger errors. This indicates that RMSE is much more useful when large errors are present and they drastically affect the model's performance. It avoids taking the absolute value of the error and this trait is useful in many mathematical calculations. In this metric also, lower the value, better is the performance of the model.