#!/usr/bin/env python
# coding: utf-8

# # Diabetes Analysis - Machine Learning End-to-End Project
# 
# This lab is a guided project that will walk you through the process of building a machine learning model to predict whether the progression of diabetes is likely to occur in a patient based on a set of diagnostic measurements. (I'm not a doctor, so I can't tell you what the measurements mean, but you can read more about them [here](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html).)
# 
# To a very large extent, this lab follows the module video lectures. If you're stuck at any point, make sure to check out the video lectures for guidance.
# 
# **Objectives**
# - Practice building a machine learning project from start to finish
# 
# **Emojis Legend**
# - 👨🏻‍💻 - Instructions; Tells you about something specific you need to do.
# - 🦉 - Tips; Will tell you about some hints, tips and best practices
# - 📜 - Documentations; provides links to documentations
# - 🚩 - Checkpoint; marks a good spot for you to commit your code to git
# - 🕵️ - Tester; Don't modify code blocks starting with this emoji

# ## Setup
# First, let's import a few common modules, ensure `MatplotLib` plots figures inline. We also ensure that you have the correct version of Python (3.10) installed.
# 
# - **Task 👨🏻‍💻**: Keep coming back to update this cell as you need to import new packages.
# - **Task 👨🏻‍💻**: Check what's already been imported here

# In[ ]:


# Python ≥3.10 is required
import sys
assert sys.version_info >= (3, 10)

# Common imports
import numpy as np
import pandas as pd
import os

# Scikit Learn imports
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use("bmh")

# to make this notebook's output stable across runs
np.random.seed(42)


# ## Diabetes Dataset
# 
# ### 1️⃣ Ask
# 
# This dataset contains several medical predictor variables and one target variable. Predictor Variables include ten baseline variables: age, sex, body mass index, average blood pressure, and six blood serum measurements (more details below) were obtained for each of n = 442 diabetes patients, as well as a quantitative measure of disease progression one year after baseline (target/outcome).
# 
# | Attribute | Description                                                                           |
# | --------- | ------------------------------------------------------------------------------------- |
# | age       | age in years                                                                          |
# | sex       | 1 - Male, 2- Female                                                                   |
# | bmi       | body mass index                                                                       |
# | bp        | average blood pressure                                                                |
# | s1        | tc, total serum cholesterol                                                           |
# | s2        | ldl, low-density lipoproteins                                                         |
# | s3        | hdl, high-density lipoproteins                                                        |
# | s4        | tch, total cholesterol / HDL                                                          |
# | s5        | ltg, possibly log of serum triglycerides level                                        |
# | s6        | glu, blood sugar level                                                                |
# | s6        | measure of the disease progression after one year of measuring the baseline variables |
# 
# The project objective is to develop a model that can predict the disease progression based on the above variables.
# 

# ### 2️⃣ Prepare
# Here we will load the dataset and split it into training and test sets. We will also perform some Exploratory Data Analysis to get some insights about the processing steps we'll need to take.

# **Task 👨🏻‍💻** : Load the dataset from `data/diabetes.csv` and store it in a variable called `diabetes`

# In[ ]:





# We need to learn about the composition of the dataset. Let's look at the first few rows of the dataset.
# 
# **Task 👨🏻‍💻** : Use the `head()` method to look at the first few rows of the dataset.
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/dataset.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# We need to know the number of rows and columns in the dataset. Let's use the `.shape` attribute to find out.
# 
# **Task 👨🏻‍💻:** Use the `.shape` attribute to find out the number of rows and columns in the dataset.
# 
# *Hint 🦉 :* 
# - `.shape` is an attribute not a method/function; you don't add `()` at the end of it to execute it.
# - `.shape` It returns a tuple of the form `(rows, columns)`

# In[ ]:





# We need to check if we have any missing values in the dataset. Let's use the `isnull()` method to find out.
# 
# **Task 👨🏻‍💻** : Use the `isnull()` method to find out if there are any missing values in the dataset.
# 
# *Hint 🦉 :* 
# - `isnull()` returns a dataframe of the same shape as the original dataframe with boolean values. `True` indicates a missing value and `False` indicates a non-missing value.
# - because it would be difficult to read a dataframe of boolean values, we can use the `sum()` method to get the total number of missing values in each column. `sum()` returns a series with the column names as the index and the total number of missing values in each column as the values.
#   - We've done this in a previous lab, so you can refer to that if you need to.
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="100" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/missing_fields.png" />
# </details>
# 

# In[ ]:





# We could also use the `info()` method to get a summary of the dataset. Let's use it to find out more about the dataset.
# 
# **Task 👨🏻‍💻** : Use the `info()` method to get a summary of the dataset.
# 
# *Hint 🦉 :* 
# - `info()` returns a summary of the dataset. 
# - It includes the number of rows and columns, the number of **non-missing values** in each column, the data type of each column and the memory usage of the dataframe.

# In[ ]:





# We could also use the `describe()` method to get a summary of the dataset. Let's use it to find out more about the dataset.
# 
# **Task 👨🏻‍💻** : Use the `describe()` method to get a summary of the dataset.

# In[ ]:





# > 🚩 : Make a git commit here

# One of the ways to get a better understanding of the dataset is to look at the distribution of the values in each column. Let's use the `describe()` method to get a summary of the distribution of values in each column.
# 
# **Task 👨🏻‍💻** : Use the `.hist()` method to plot a histogram of each column in the dataset.
# 
# *Hint 🦉 :* 
# - use the `figsize` parameter to set the size of the figure to `(20, 15)`
# - use the `bins` parameter to set the number of bins to `50`
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="800" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/diabetes_hist.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# #### ☕️ Coffee Break
# Here are some observations we can make from the what we've done so far (usually I would ask you to make these observations):
# - The dataset has 442 rows and 11 columns. 
#   - n = 422
#   - m = 10
#   - outcome/target = 1
# - There are is a single missing feature for a single row in the dataset. (I purposefully removed it from the dataset so we can practice imputing missing values)
# - the `SEX` column has only two unique values: 1 and 2. This means that the column is categorical and not numerical.
#   - We will need to replace the values with `MALE` and `FEMALE` instead of `1` and `2`.
#   - We will need to use the `OneHotEncoder` to encode this column.
# - The values in each column are on a different scales. Some ranging from 20 to 80, others from 100 to 400, and so on. We will need to scale the values before we can use them to train a machine learning model.
# 

# #### EDA
# Let's perform some more data exploration. Let's visualize the relationship between the features and the target.

# **Task 👨🏻‍💻** : Use the `scatter_matrix()` method from the `pandas.plotting` module to plot a scatter plot of each column in the dataset against the target column.
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/scatter_matrix.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# **✨ Extra Credit Task 👨🏻‍💻**: <u>For 1 point of Extra Credit:</u> plot the same chart again but this time without the `SEX` column.
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/scatter_matrix_no_sex.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# Let's explore the correlation between the target variable and the the input features
# 
# **Task 👨🏻‍💻** : Use the `corr()` method to print out the correlation matrix for the `Y` column in this dataset. Sort the values in descending order.
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="150" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/Y_corr.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻** : what are the top 3 features that are most correlated with the target column? 
Answer the question in this cell.
# > 🚩 : Make a git commit here

# **✨ Extra Credit Task 👨🏻‍💻**: 
# <u>For 5 Extra Credit Points:</u> Plot the correlation matrix using a heatmap. You can use the `seaborn` module to do this. You can refer to the [documentation](https://seaborn.pydata.org/generated/seaborn.heatmap.html) for more information.
# 
# *Hint 🦉 :* 
# - use the correlation matrix generated in the previous step to generate the heatmap.
# - use the `coolwarm` color map to get a similar color scheme to the example.
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/corr_heatmap.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# #### Split the dataset into training and test sets
# Now before we go any further, we need to split the dataset we have into two parts:
# - a training set
# - a test set
# 
# This step is important because we need to train our model, then test it against some data that it hasn't seen before. If we don't do this, we won't be able to tell if our model is overfitting or not.

# **Task 👨🏻‍💻** : Use the `train_test_split()` function from the `sklearn.model_selection` package to split the dataset into a training set named `random_train_set` and a test set and `random_test_set`.
# 
# *Hint 🦉 :* 
# - use the `random_state` parameter to set the random seed to `42` - this will ensure that we get the same results every time we run the code.
# - use the `test_size` parameter to set the size of the test set to `0.2`; the test set is 20% of the size of the dataset.

# In[ ]:





# > 🚩 : Make a git commit here

# Now as mentioned in the lectures, a random split of the data may cause issues of bias. Where one subset of the data maybe overly represented in the training set. This leads misrepresenting model of the data. 
# 
# To avoid this, we can use stratified sampling to ensure that the training set and test set are representative of the original dataset. Usually, a Subject Matter Expert (SME) would be able to tell us which column to use for stratified sampling. 
# 
# In this case, we will use the `SEX` column.
# 
# **Task 👨🏻‍💻** : Use the `StratifiedShuffleSplit()` function from the `sklearn.model_selection` package to split the dataset into a training set named `sex_strat_train_set` and a test set and `sex_strat_test_set`. Split the data on the `SEX` column.
# 
# *Hint 🦉 :*
# - use the `random_state` parameter to set the random seed to `42` - this will ensure that we get the same results every time we run the code. it will also ensure you get similar values to the ones in the screenshots provided in this assignment.
# - use the `test_size` parameter to set the size of the test set to `0.2`; the test set is 20% of the size of the dataset.

# In[ ]:





# > 🚩 : Make a git commit here

# The `SEX` column was an okay choice to stratify the sampling on. However, we can also try to use the `AGE` column. However, we'll need to introduce a categorical representation of the `AGE` column by binning the values into different age groups.
# 
# **Task 👨🏻‍💻** : Create a new column called `AGE_CAT` and bin the values in the `AGE` column into the following categories:
# - 10 - 30
# - 30 - 50
# - 50 - 70
# - 70 - 90
# 
# *Hint 🦉 :* 
# - Use the `pd.cut()` function to bin the values in the `AGE` column into the categories above.

# In[ ]:





# show the distribution of the values in the `AGE_CAT` column. using the `hist()` method to show a diagram of the distribution of the values in the `AGE_CAT` column.
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/age_cat_hist.png" />
# </details>

# In[ ]:





# **Task 👨🏻‍💻** : Use the `StratifiedShuffleSplit()` function from the `sklearn.model_selection` package to split the dataset into a training set named `age_sex_strat_train_set` and a test set and `age_sex_strat_test_set`. Split the data on the `AGE_CAT` column.
# 
# *Hint 🦉 :*
# - use the `random_state` parameter to set the random seed to `42` - this will ensure that we get the same results every time we run the code. it will also ensure you get similar values to the ones in the screenshots provided in this assignment.
# - use the `test_size` parameter to set the size of the test set to `0.2`; the test set is 20% of the size of the dataset.

# In[ ]:





# > 🚩 : Make a git commit here

# We will use the age-split sets for the rest of the lab. But we need to remember to drop the `AGE_CAT` column from the training set and test set.
# 
# *Note 🦉:* This doesn't mean that splitting the data by the `SEX` column is wrong. It all depends on the problem you're trying to solve and the data you have. This is usually guided by the Subject Matter Expert (SME).

# **Task 👨🏻‍💻** : Drop the `AGE_CAT` column from the training set and test set.
# 
# *Hint 🦉 :* 
# - use the `drop()` method to drop the `AGE_CAT` column from the training set and test set.
# - you'll need to call it twice, once for the training set `age_sex_strat_train_set` and once for the test set `age_sex_strat_test_set`.
# - You can use the `inplace` parameter to drop the column in place.

# In[ ]:





# > 🚩 : Make a git commit here

# #### Separate the features and labels
# Let's separate the features `X` from the labels `y`. We'll use the training set for this.

# **Task 👨🏻‍💻** : Create a copy of the training set <u>without</u> the output `y` and store it in a variable called `diabetes_X`, and create a copy of the dataset with <u>Only</u> the column `Y` and name it `diabetes_y`.

# In[ ]:





# <details>
#   <summary>Running the following cell, should produce an output that looks like this: (🦉 click me)</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/prepare-done1.png" />
#   <br />
#   <strong>OR</strong>
#   <br />
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/prepare-done2.png" />
# </details>

# In[ ]:


display(diabetes_X.head())
display(diabetes_y.head())


# > 🚩 : Make a git commit here

# ### 🏖 ☕️ Take a break here
# make sure you understand what we've done so far.

# ### 3️⃣ Process
# In this section, we'll process and clean it in preparation for the model creation and analysis work the data. 
# 
# Here are some of what we will do:
# - impute missing values (numerical data)
# - scale numerical features (numerical data)
# - encode categorical features (categorical data)
# - combining features into new features
# 
# We will also compose all of these steps into a single pipeline.
# 
# Let's start doing them individually.

# #### Impute Missing Values
# We'll use the `SimpleImputer` class from the `sklearn.impute` package to impute the missing values in the dataset. I, purposefully, removed some values from the dataset so that we can practice imputing missing values.
# 
# Keep in mind, even if there was no missing values in the dataset, we would still need to implement the imputer. Why? Because we need to make sure that our processing pipeline can be applied to new data that <u>may</u> have missing values.

# Since the imputer applies to <u>numerical</u> features, we'll need to create a copy of the dataset with only the numerical features, and another with only the categorical features.
# 
# **Task 👨🏻‍💻** : Create a copy of the dataset with only the numerical features and name it `diabetes_X_num`. Create another copy of the dataset with only the categorical features and name it `diabetes_X_cat`. 
# 
# *Hint:*
# - MAKE SURE you start with the diabetes_X dataframe that we created earlier. not the full dataset.

# In[ ]:





# <details>
#   <summary>Running the following cell, should produce an output that looks like this: (🦉 click me)</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/num-cat.png" />
# </details>

# In[ ]:


display(diabetes_X_num.head())
display(diabetes_X_cat.head())


# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻** : Create an instance of the `SimpleImputer` class and name it `imputer`. Use the `mean` value as the imputer strategy.
# 
# *Hint 🦉 :* 
# - use the `diabetes_X_num` dataframe to create the imputer.
# - set the `strategy` parameter to `mean`
# - name the transformed dataset `diabetes_X_num_imputed`

# In[ ]:





# Keep in mind, all the `sklearn` transforms return a numpy array. if you want to print the data such that we can see them with the column names as we're used to in `pandas`, we need to convert that back to a DataFrame.
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/data-imputed.png" />
# </details>

# In[ ]:


# assuming the imputer has been fitted and transformed and the result is stored in X
diabetes_X_num_imputed = pd.DataFrame(diabetes_X_num_imputed, columns=diabetes_X_num.columns, index=diabetes_X_num.index)
diabetes_X_num_imputed.head()


# > 🚩 : Make a git commit here

# #### Scaling and Normalizing Numerical Features
# <img width="500" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/diabetes_hist.png" />
# 
# As you can see from the histograms and the `describe()` output (if you do it), the values in each column are on different scales. This leads the machine learning algorithms giving more weight to the features. We need to scale the values in each column to the same scale. 

# **Task 👨🏻‍💻** : use the `StandardScaler` class to transform/scale the values in each column to the same scale.
# 
# *Hint 🦉 :* 
# - name the transformed data `diabetes_X_num_scaled`
# - make sure you use the `diabetes_X_num_imputed` as input to the scaler

# In[ ]:





# Similar to we've done above after the imputer, we need to convert the numpy array to a DataFrame.
# 
# **Task 👨🏻‍💻** convert the numpy array `diabetes_X_num_scaled` back to a pandas DataFrame. 
# 
# *Hint 🦉 :* Look above 

# In[ ]:





# **Task 👨🏻‍💻** : print the first 5 records of the  use the `diabetes_X_num_scaled`
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/data-scaled.png" />
# </details>
# 

# In[ ]:


diabetes_X_num_scaled.head()


# > 🚩 : Make a git commit here

# #### Encode Categorical Features
# Let's now process and transform the categorical features. In the videos we mentioned 2 types of categorical feature encoders: `OrdinalEncoder` and `OneHotEncoder`. We'll use the `OneHotEncoder` to encode the categorical features in the dataset.
# 
# **Task 👨🏻‍💻** : Why don't we use the `OrdinalEncoder` to encode the categorical features in the dataset?
👨🏻‍💻 Answer the question in this cell:

# > 🚩 : Make a git commit here

# Let's see how many records exist for each category in the `SEX` column.

# In[ ]:


diabetes_X_cat['SEX'].value_counts()


# If we encode the data with the current values, the category names would be `1` and `2`.
# 
# Let's change the values to `male` and `female`.
# 
# **Task 👨🏻‍💻** : Use the `replace()` method to replace the values in the `SEX` column with `male` and `female`. 
# 
# (done for you; make sure you understand what's happening)

# In[ ]:


diabetes_X_cat['SEX'].replace({1: 'MALE', 2: 'FEMALE'}, inplace=True)
display(diabetes_X_cat.head())


# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻** : Encode the `SEX` column using the `OneHotEncoder` class. and print the `categories_` attribute of the encoder.
# 
# *Hint 🦉 :* 
# - Create an instance of the `OneHotEncoder` class and name it `cat_encoder`.
# - use the `diabetes_X_cat` dataframe to `fit` the encoder.
# - the results of this transformation will be a scipy sparse matrix. this is a more memory efficient way of storing the data. (reference the lecture videos for more info)
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="400" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/encoder-cats.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# #### Pipelines
# We've seen how to process the data in individual steps. But we need to combine all the steps into a single pipeline. This will make it easier to apply the same transformations to the test set, to new data, and to any other data that we may have in the future.
# 
# We'll start by creating a pipeline for the numerical features. The numerical pipeline will impute the missing values, and scale the values.
# 
# **Task 👨🏻‍💻** : Create a pipeline for the numerical features that will impute the missing values, and scale the values in each column to the same scale. Name the pipeline `num_pipeline`.

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻** : Create a pipeline for the categorical features that will encode categorical columns using the `OneHotEncoder` Transformer. Name the pipeline `cat_pipeline`.

# In[ ]:





# > 🚩 : Make a git commit here

# To combine the numerical and categorical pipelines, we'll use the `ColumnTransformer` class from the `sklearn.compose` package. This ColumnTransformer allows us to apply different transformation pipelines to different columns.
# 
# **Task 👨🏻‍💻** : Create a `ColumnTransformer` that will apply the `num_pipeline` to the numerical columns, and the `cat_pipeline` to the categorical columns. Name the ColumnTransformer `full_pipeline`.
# 
# *Hint 🦉 :* 
# - create 2 lists/arrays of the column names for each subset. one for the numerical columns, and one for the categorical columns.
# - name the transformed dataset `diabetes_X_prepared`

# In[ ]:





# > 🚩 : Make a git commit here

# Again, any output of the `sklearn` transforms is a numpy array. If we want to get the first 5 records of the transformed dataset, we need to do the following:
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/diabetes-assignment/prepared-numpy.png" />
# </details>

# In[ ]:


diabetes_X_prepared[:5]


# if we want to convert the numpy array to a DataFrame, we can do the following:
# 
# - note: the column names are not in the same order as the original dataset. this is because the `OneHotEncoder` class returns the encoded columns in alphabetical order.

# In[ ]:


# Converting the numpy array to a pandas dataframe
all_columns = diabetes_X_num.columns.tolist() + cat_encoder.categories_[0].tolist()


diabetes_X_prepared = pd.DataFrame(diabetes_X_prepared, columns=all_columns, index=diabetes_X_num.index)


# **For 5 Extra Credit Points:**
# - Create a custom transformer that would a new feature that is the ratio of the `BMI` to the `AGE`. Add this transformer to the `full_pipeline` and make sure it works.

# In[ ]:





# > 🚩 : Make a git commit here (if you've done the extra credit)

# 🏖 ☕️ Take a break here and make sure you understand what we've done so far.

# ### 4️⃣ Analyze
# In this section, we'll train 2 machine learning models to make predictions about the diabetes progression after 1 year, given a number of predictors.
# 
# At this stage we should have the following 4 datasets:
# - `diabetes_X_prepared` - the inputs for the training set
# - `diabetes_y` - the outputs for the training set
# - `age_sex_strat_test_set` - the test set (both X and y)

# In[ ]:


# ⛔️ Do not uncomment this cell. This is what I used to save a copy of the prepared data to csv files.

# diabetes_X_prepared.to_csv('data/train_X_prepared.csv', index=False)
# diabetes_y.to_csv('data/trainy.csv', index=False)

# age_sex_strat_test_set.to_csv('data/test_X_y.csv', index=False)


# Now in case you didn't get the prepare and process steps right, you can use the `diabetes_X_prepared` and `diabetes_y` datasets that I've created for you.
# 
# uncomment the following cell and execute it.

# In[ ]:


# diabetes_X_prepared = pd.read_csv("data/train_X_prepared.csv")
# diabetes_y = pd.read_csv("data/train_y.csv")
# age_sex_strat_test_set = pd.read_csv("data/test_X_y.csv")


# Let's build our linear regression model. We'll use the `LinearRegression` class from the `sklearn.linear_model` package.
# 
# **Task 👨🏻‍💻** : Create an instance of the `LinearRegression` class and name it `lin_reg`.
# 
# *Hint*:
# - the input is the `diabetes_X_prepared` dataset, the output is the `diabetes_y` dataset

# In[ ]:





# to test how well the model did, we'll calculate the RMSE on the training set

# In[ ]:





# > 🚩 : Make a git commit here

# Let's create a polynomial regression model. We'll use the `PolynomialFeatures` class from the `sklearn.preprocessing` package to create a new dataset with the polynomial features.
# 
# **Task 👨🏻‍💻** : 
# - Create an instance of the `PolynomialFeatures` class and name it `poly_features`. Set the `degree` parameter to `2`.
# - Create a new linear regression model and name it `poly_reg` and train it using the `poly_features`.
# 
# *Hint*:
# - you can do this in 2 steps, or you can build a pipeline that will do both steps in one go.

# In[ ]:





# to test how well the model did, we'll calculate the RMSE on the training set

# In[ ]:





# > 🚩 : Make a git commit here

# Given the following features, predict the diabetes progression after 1 year (Y) using the linear regression and polynomial regression models.
# 
# Features:
# - Age:50
# - Sex: female (2)
# - BMI: 26.2
# - BP: 97
# - S1: 186
# - S2: 105.4
# - S3: 49
# - S4: 4
# - S5: 5.0626
# - S6: 88
# 
# **Task 👨🏻‍💻** : Given the above features, predict the diabetes progression after 1 year (Y) using the linear regression and polynomial regression models.
# 
# *Hint 🦉 :* 
#  - you'll need to create a list/array/DataFrame with the above features.
#  - you will need to transform that input using the `full_pipeline` before passing it to the model.
#  - you can use the `predict()` method to make predictions
#    - linear regression model gives a prediction of 165.07515659
#    - polynomial regression model gives a prediction of 156.5625
#  - At this point, the model is trained, and the processing pipelines are trained. Make sure you don't re-train the model again with the input.
#    - in other words, you should only use `transform()` calls not the `fit_transform()` and certainly not the `fit()`.

# In[ ]:





# > 🚩 : Make a git commit here

# ## Evaluate against the test set
# Let's evaluate the models using the test set.
# 
# **Task 👨🏻‍💻** : Calculate the RMSE on the test set for the linear regression model **AND** the polynomial regression model.
# 
# *Hint*:
# - remember to separate the X and y from the test set
# - remember to transform the X using the `full_pipeline`
# - I left some print statements. You don't need to change them. Use their output to answer the questions below.

# In[ ]:





# In[ ]:


# Don't modify the code below this line
print("-----------------------------------------------------------------")
print(f"Linear Regression RMSE on the training set: {lin_rmse}")
print(f"Linear Regression RMSE on the test set: {test_lin_rmse}")
print("-----------------------------------------------------------------")
print(f"Polynomial Regression RMSE on the training set: {poly_rmse}")
print(f"Polynomial Regression RMSE on the test set: {test_poly_rmse}")
print("-----------------------------------------------------------------")


# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻** :Comparing the RMSE values for the linear Regression and polynomial regression models on the training and test sets, which model performed better?
# 
# Here's the output I received. You should get similar results.
# 
# | Model                 | RMSE on Training Set | RMSE on Test Set  |
# | --------------------- | -------------------- | ----------------- |
# | Linear Regression     | 53.63500319865106    | 53.11536987833757 |
# | Polynomial Regression | 47.80449635259552    | 54.2584162004519  |
Answer the question in this cell:

# > 🚩 : Make a git commit here

# ## Wrap up
# - **Task 👨🏻‍💻** : Remember to update the self reflection and self evaluations on the `README` file.
# - **Task 👨🏻‍💻** : Make sure you run the following cell. It converts this Jupyter notebook to a Python script. This allows me to provide feedback on your code.
# 

# In[ ]:


get_ipython().system('jupyter nbconvert --to python diabetes-analysis.ipynb')


# > 🚩 : Make a git commit here
