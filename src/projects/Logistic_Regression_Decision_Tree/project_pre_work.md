### Review notebooks 

### Bootcamp
* 01-Logistic Regression with Python.ipynb - `Completed`
* 03-Logistic Regression Project - Solutions.ipynb - `Completed`
* 01-Decision Trees and Random Forests in Python.ipynb - `Completed`
* 03-Decision Trees and Random Forest Project - Solutions.ipynb - `Completed`

## GL 

* 01-Logistic Regression with Python.ipynb - `Completed`
* 03-Logistic Regression Project - Solutions.ipynb - `Completed`
* Additional_CaseStudy_Logistic_AIML.ipynb - `Completed`
* IncomeGroupClassification_CaseStudy_AIML.ipynb - `Completed`
* Logistic Regression - Hands On.ipynb `Completed`

## GL Decision Tree

* DecisionTree_Notebook (1).ipynb - `Completed`
* Additional_CaseStudy_Loan_Delinquent-AIML.ipynb - `Completed`
* OnlineShoppersPurchasing_Intention

# Project Flow 

### TBD


## Context 
DeltaSquare is an NGO that works with the Government on matters of social policy to bring about a change in the lives of underprivileged sections of society. They are given a task of coming up with a policy framework by analyzing a dataset that the government received from WHO. You as a data scientist at DeltaSquare are tasked with solving this problem and sharing a proposal for the government. 

### Objective

1. What are the different factors that influence the income of an individual?

2. To build a prediction model that can help the government formulate policies for the right pockets of the society.


### Dataset

The data contains characteristics of the people

* age: continuous - age of a Person 
* workclass: Where does a person works - categorical -Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
* fnlwgt: continuous - Weight assigned by Current Population Survey (CPS) - People with similar demographic characteristics should have similar weights since it is a feature aimed to allocate similar weights to people with similar demographic characteristics.
* education: Degree the person has - Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
* education-num: no. of years a person studied - continuous.
* marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
* occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
* race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
* sex: Female, Male.
* capital-gain: Investment gain of the person other than salary - continuous
* capital-loss: Loss from investments - continuous
* hours-per-week: No. of hours a person works - continuous.
* native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinidad&Tobago, Peru, Hong, Holand-Netherlands.
* salary: >50K, <=50K (dependent variable, the salary is in Dollars per year)

## Import Libraries
`Let's import some libraries to get started!`

```
# this will help in making the Python code more structured automatically (good coding practice)
%load_ext nb_black

import warnings

warnings.filterwarnings("ignore")

# Libraries to help with reading and manipulating data

import pandas as pd
import numpy as np

# Library to split data
from sklearn.model_selection import train_test_split

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)


# To build model for prediction

from sklearn.linear_model import LogisticRegression

# To get diferent metric scores
# To get diferent metric scores

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    plot_confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
```

## The Data
`Let's start by reading in the csv file into a pandas dataframe.`

### shape 
data.shape



### Let's check the duplicate data. And if any, we should remove it.

### Let's drop the duplicate values¶
## Check the data types of the columns for the dataset.
### info 


```
- There are total 10 columns and 1,000 observations in the dataset
- We have only three continuous variables - Age, Credit Amount, and Duration.
- All other variables are categorical.
- There no missing values in the dataset.
```

### Summary of the data

data.describe(include=["object", "bool"])


data.describe().T

```




**Observations**

- Mean value for the age column is approx 35 and the median is 33. This shows that majority of the customers are under 35 years of age.
- Mean amount of credit is approx 3,271 but it has a wide range with values from 250 to 18,424. We will explore this further in univariate analysis.
- Mean duration for which the credit is given is approx 21 months.

* `age`: Average age of people in the dataset is 38 years, age has a wide range from 17 to 90 years.
* `education_no_of_years`: The average education in years is 10 years. There's a large difference between the minimum value and 25th percentile which indicates that there might be outliers present in this variable.
* `capital_gain`: There's a huge difference in the 75th percentile and maximum value of capital_gain indicating the presence of outliers. Also, 75% of the observations are 0.
* `capital_loss`: Same as capital gain there's a huge difference in the 75th percentile and maximum value indicating the presence of outliers. Also, 75% of the observations are 0.
* `working_hours_per_week`: On average people work for 40 hours a week. A vast difference in minimum value and 25th percentile, as well as 75th percentile and the maximum value, indicates that there might be outliers present in the variable.

```

### data insights

### think about it 

### missing values 

### value_counts of category columns 

```
# Making a list of all catrgorical variables
cat_col = [
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose",
    "Risk",
]

# Printing number of count of each unique value in each column
for column in cat_col:
    print(data[column].value_counts())
    print("-" * 40)
```

```
- We have more male customers as compared to female customers
- There are very few observations i.e., only 22 for customers with job category - unskilled and non-resident
- We can see that the distribution of classes in the target variable is imbalanced i.e., only 30% observations with defaulters.
- Most of the customers are not at risk.
```

### range of outliers on numerical columns

## Calculate diabetes ratio of True/False from outcome variable 

```
n_true = len(pdata.loc[pdata['class'] == True])
n_false = len(pdata.loc[pdata['class'] == False])
print("Number of true cases: {0} ({1:2.2f}%)".format(n_true, (n_true / (n_true + n_false)) * 100 ))
print("Number of false cases: {0} ({1:2.2f}%)".format(n_false, (n_false / (n_true + n_false)) * 100))


So we have 34.90% people in current data set who have diabetes and rest of 65.10% doesn't have diabetes. 

Its a good distribution True/False cases of diabetes in data.

```


### data conversions if any 
### Treating Outliers


### Treating missing values 



# Exploratory Data Analysis
`Let's begin some exploratory data analysis! We'll start by checking out missing data!`

### count plot with stack

```
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
```

### joint plot 

sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')

### lm plot 

plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')

### Treating Outliers



## Identify Correlation in data 

pdata.corr() # It will show correlation matrix 

```
### However we want to see correlation in graphical representation so below is function for that
def plot_corr(df, size=11):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    for (i, j), z in np.ndenumerate(corr):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')


In above plot yellow colour represents maximum correlation and blue colour represents minimum correlation.
We can see none of variable have correlation with any other variables.

```

## Missing Data
`We can use seaborn to create a simple heatmap to see where we are missing data!`

`sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')`

### Count data on target column 


```
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


```
## Univariate analysis
### Dist Plot 

`sns.histplot(df['Age'].dropna(),kde=False,color='darkred',bins=20)`
### labeled_barplot with %

### distribution_plot_wrt_target

# BI variate 

### stacked_barplot

### join plot 

```sns.jointplot(x='Age',y='Area Income',data=ad_data)```

```sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde');```

## pair plot 

sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')


## Data Cleaning

`We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
However we can be smarter about this and check the average age by passenger class. For example:`

```
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
```

`impute missing values using avg - impute_age`

or

### Replace 0s with serial mean 

### DROP Columns with missing values 


## Converting Categorical Features 

`We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.`



Great! Our data is ready for our model!


# model 

```

### Model evaluation criterion

### Model can make wrong predictions as:

1. Predicting a customer is not going to default but in reality the customer will default - Loss of resources
2. Predicting a customer is going to default but in reality the customer will not default - Loss of opportunity


### Which Loss is greater ? 

* Loss of opportunity will be the greater loss as the bank will be losing a potential customer.

### How to reduce this loss i.e need to reduce False Negatives ?

* Company would want to reduce false negatives, this can be done by maximizing the Recall. Greater the recall lesser the chances of false negatives.


#### First, let's create functions to calculate different metrics and confusion matrix so that we don't have to use the same code repeatedly for each model.
* The model_performance_classification_sklearn_with_threshold function will be used to check the model performance of models. 
* The confusion_matrix_sklearn_with_threshold function will be used to plot confusion matrix.

```

# Building a Logistic Regression model

Let's start by splitting our data into a training set and test set.
Now it's time to do a train test split, and train our model!

## Train Test Split

#### Spliting the data 
We will use 70% of data for training and 30% for testing.

** Split the data into training set and testing set using train_test_split**

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1), 
                                                    df['Survived'], test_size=0.30, 
                                                    random_state=101)

```

### Lets check split of data in %
```
print("{0:0.2f}% data is in training set".format((len(x_train)/len(pdata.index)) * 100))
print("{0:0.2f}% data is in test set".format((len(x_test)/len(pdata.index)) * 100))
```

```
Now lets check diabetes True/False ratio in split data 

print("Original Diabetes True Values    : {0} ({1:0.2f}%)".format(len(pdata.loc[pdata['class'] == 1]), (len(pdata.loc[pdata['class'] == 1])/len(pdata.index)) * 100))
print("Original Diabetes False Values   : {0} ({1:0.2f}%)".format(len(pdata.loc[pdata['class'] == 0]), (len(pdata.loc[pdata['class'] == 0])/len(pdata.index)) * 100))
print("")
print("Training Diabetes True Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))
print("Training Diabetes False Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))
print("")
print("Test Diabetes True Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))
print("Test Diabetes False Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))
print("")

```

## Training and Predicting

`from sklearn.linear_model import LogisticRegression`

```
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
```

`Let's move on to evaluate our model!`

### intercept & coef

```
coef_df = pd.DataFrame(model.coef_)
coef_df['intercept'] = model.intercept_
print(coef_df)
```
## Scoring our Decision Tree
## Predictions and Evaluations
** Now predict values for the testing data.**

`predictions = logmodel.predict(X_test)`

** Create a classification report for the model.**

`We can check precision,recall,f1-score using classification report!`
`from sklearn.metrics import classification_report`
`print(classification_report(y_test,predictions))`

### confusion matrix in graph, make_confusion_matrix method, get_recall_score

```
cm=metrics.confusion_matrix(y_test, y_predict, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["Actual 1"," Actual 0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True,fmt='g')
plt.show()


The confusion matrix

True Positives (TP): we correctly predicted that they do have diabetes 50

True Negatives (TN): we correctly predicted that they don't have diabetes 131

False Positives (FP): we incorrectly predicted that they do have diabetes (a "Type I error") 15 Falsely predict positive Type I error

False Negatives (FN): we incorrectly predicted that they don't have diabetes (a "Type II error") 35 Falsely predict negative Type II error

```
```

* The ratio of positives to negatives is 3:7, so if our model marks each sample as negative, then also we'll get 70% accuracy, hence accuracy is not a good metric to evaluate here.

**What does a bank want?**
* A bank wants to minimize the loss - it can face 2 types of losses here: 
   * Whenever bank lends money to a customer, they don't return that.
   * A bank doesn't lend money to a customer thinking he will default but in reality he won't - oppurtunity loss.

**Which loss is greater ?**
* Customer not returning the money back.

**Since we don't want people to default on the loans we should use Recall as a metric of model evaluation instead of accuracy.**

* Recall - It gives the ratio of True positives to Actual positives, so high Recall implies low false negatives, i.e. low chances of predicting a defaulter as non defaulter
```

## different models with different thresholds

#### ROC-AUC
### logit_roc_auc_train and on test 
### Optimal threshold using AUC-ROC curve
#### Let's use Precision-Recall curve and see if we can find a better threshold


## Sequential Feature Selector 



### Model Performance Summary

# imporant features using importances horizontal bar chat 

## Decision Trees

We'll start just by training a single decision tree.

## Prediction and Evaluation 

Let's evaluate our decision tree.

## Tree Visualization

Scikit learn actually has some built-in visualization capabilities for decision trees, you won't use this often and it requires you to install the pydot library, but here is an example of what it looks like and the code to execute this:

# tree  Text report

## Reducing over fitting with max depth - with  Prediction and Evaluation 
### Using GridSearch for Hyperparameter tuning of our tree model  - with  Prediction and Evaluation 

# Total impurity of leaves vs effective alphas of pruned tree

# Accuracy vs alpha for training and testing sets


## Random Forests

Now let's compare the decision tree model to a random forest.

## Prediction and Evaluation 

Let's evaluate our decision tree.

# compare all the models

### Conclusion

```
* By changing the threshold of the logistic regression model we were able to see a significant improvement in the model performance.
* The model achieved a recall of 0.58 on the training set with threshold set at 0.32.

```

## Recommendations

- From our logistic regression model we identified that Duration is a significant predictor of a customer being a defaulter. 
- Bank should target more male customers as they have lesser odds of defaulting.
- We saw in our analysis that customers with a little or moderate amount in saving or checking accounts are more likely to default.  The bank can be more strict with its rules or interest rates to compensate for the risk.
- We saw that customers who have rented or free housing are more likely to default. The bank should keep more details about such customers like hometown addresses, etc. to be able to track them.
- Our analysis showed that younger customers are slightly more likely to default. The bank can alter its policies to deal with this.