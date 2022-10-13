# Multiclass Classification on the Body Performance Dataset
```python
from random import randint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
from sklearn.metrics import confusion_matrix
```

## The Data

This dataset contains age, gender (M or F), and a set of fitness indicators such as systolic and diastolic blood pressure. THat is all followed by an overall 'class' that is either A (best), B, C, or D (worst).
It can be found [here]https://www.kaggle.com/datasets/kukuroo3/body-performance-data.


```python
body_performance_data = pd.read_csv("bodyPerformance.csv")
body_performance_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
    
    table {
        overflow: auto;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>height_cm</th>
      <th>weight_kg</th>
      <th>body fat_%</th>
      <th>diastolic</th>
      <th>systolic</th>
      <th>gripForce</th>
      <th>sit and bend forward_cm</th>
      <th>sit-ups counts</th>
      <th>broad jump_cm</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27.0</td>
      <td>M</td>
      <td>172.3</td>
      <td>75.24</td>
      <td>21.3</td>
      <td>80.0</td>
      <td>130.0</td>
      <td>54.9</td>
      <td>18.4</td>
      <td>60.0</td>
      <td>217.0</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25.0</td>
      <td>M</td>
      <td>165.0</td>
      <td>55.80</td>
      <td>15.7</td>
      <td>77.0</td>
      <td>126.0</td>
      <td>36.4</td>
      <td>16.3</td>
      <td>53.0</td>
      <td>229.0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31.0</td>
      <td>M</td>
      <td>179.6</td>
      <td>78.00</td>
      <td>20.1</td>
      <td>92.0</td>
      <td>152.0</td>
      <td>44.8</td>
      <td>12.0</td>
      <td>49.0</td>
      <td>181.0</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32.0</td>
      <td>M</td>
      <td>174.5</td>
      <td>71.10</td>
      <td>18.4</td>
      <td>76.0</td>
      <td>147.0</td>
      <td>41.4</td>
      <td>15.2</td>
      <td>53.0</td>
      <td>219.0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28.0</td>
      <td>M</td>
      <td>173.8</td>
      <td>67.70</td>
      <td>17.1</td>
      <td>70.0</td>
      <td>127.0</td>
      <td>43.5</td>
      <td>27.1</td>
      <td>45.0</td>
      <td>217.0</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
</div>



It contains the details of 13393 different adults, with 11 predictors for the class.


```python
body_performance_data.shape
```




    (13393, 12)



No numerical variables are over 25% zeroes.


```python
body_performance_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
    
    table {
        overflow: auto;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>height_cm</th>
      <th>weight_kg</th>
      <th>body fat_%</th>
      <th>diastolic</th>
      <th>systolic</th>
      <th>gripForce</th>
      <th>sit and bend forward_cm</th>
      <th>sit-ups counts</th>
      <th>broad jump_cm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>13393.000000</td>
      <td>13393.000000</td>
      <td>13393.000000</td>
      <td>13393.000000</td>
      <td>13393.000000</td>
      <td>13393.000000</td>
      <td>13393.000000</td>
      <td>13393.000000</td>
      <td>13393.000000</td>
      <td>13393.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36.775106</td>
      <td>168.559807</td>
      <td>67.447316</td>
      <td>23.240165</td>
      <td>78.796842</td>
      <td>130.234817</td>
      <td>36.963877</td>
      <td>15.209268</td>
      <td>39.771224</td>
      <td>190.129627</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.625639</td>
      <td>8.426583</td>
      <td>11.949666</td>
      <td>7.256844</td>
      <td>10.742033</td>
      <td>14.713954</td>
      <td>10.624864</td>
      <td>8.456677</td>
      <td>14.276698</td>
      <td>39.868000</td>
    </tr>
    <tr>
      <th>min</th>
      <td>21.000000</td>
      <td>125.000000</td>
      <td>26.300000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.000000</td>
      <td>162.400000</td>
      <td>58.200000</td>
      <td>18.000000</td>
      <td>71.000000</td>
      <td>120.000000</td>
      <td>27.500000</td>
      <td>10.900000</td>
      <td>30.000000</td>
      <td>162.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>32.000000</td>
      <td>169.200000</td>
      <td>67.400000</td>
      <td>22.800000</td>
      <td>79.000000</td>
      <td>130.000000</td>
      <td>37.900000</td>
      <td>16.200000</td>
      <td>41.000000</td>
      <td>193.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>174.800000</td>
      <td>75.300000</td>
      <td>28.000000</td>
      <td>86.000000</td>
      <td>141.000000</td>
      <td>45.200000</td>
      <td>20.700000</td>
      <td>50.000000</td>
      <td>221.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>64.000000</td>
      <td>193.800000</td>
      <td>138.100000</td>
      <td>78.400000</td>
      <td>156.200000</td>
      <td>201.000000</td>
      <td>70.500000</td>
      <td>213.000000</td>
      <td>80.000000</td>
      <td>303.000000</td>
    </tr>
  </tbody>
</table>
</div>



There are a close-to-equal number of members of each type of class.


```python
body_performance_data["class"].value_counts()
```




    C    3349
    D    3349
    A    3348
    B    3347
    Name: class, dtype: int64



There are slightly more female members than male.


```python
body_performance_data["gender"].value_counts()
```




    M    8467
    F    4926
    Name: gender, dtype: int64



## Preprocessing

The target variable (the class) is separated from the predictors.


```python
features = body_performance_data.drop('class', axis = 1)
target = body_performance_data[['class']]
```

The numerical predictors are separated from the categorical features, so they can be standardised in order to make the comprison easier for the model. 


```python
numeric_features = features.drop(['gender'], axis =1)
numeric_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>height_cm</th>
      <th>weight_kg</th>
      <th>body fat_%</th>
      <th>diastolic</th>
      <th>systolic</th>
      <th>gripForce</th>
      <th>sit and bend forward_cm</th>
      <th>sit-ups counts</th>
      <th>broad jump_cm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27.0</td>
      <td>172.3</td>
      <td>75.24</td>
      <td>21.3</td>
      <td>80.0</td>
      <td>130.0</td>
      <td>54.9</td>
      <td>18.4</td>
      <td>60.0</td>
      <td>217.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25.0</td>
      <td>165.0</td>
      <td>55.80</td>
      <td>15.7</td>
      <td>77.0</td>
      <td>126.0</td>
      <td>36.4</td>
      <td>16.3</td>
      <td>53.0</td>
      <td>229.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31.0</td>
      <td>179.6</td>
      <td>78.00</td>
      <td>20.1</td>
      <td>92.0</td>
      <td>152.0</td>
      <td>44.8</td>
      <td>12.0</td>
      <td>49.0</td>
      <td>181.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32.0</td>
      <td>174.5</td>
      <td>71.10</td>
      <td>18.4</td>
      <td>76.0</td>
      <td>147.0</td>
      <td>41.4</td>
      <td>15.2</td>
      <td>53.0</td>
      <td>219.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28.0</td>
      <td>173.8</td>
      <td>67.70</td>
      <td>17.1</td>
      <td>70.0</td>
      <td>127.0</td>
      <td>43.5</td>
      <td>27.1</td>
      <td>45.0</td>
      <td>217.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
standardScaler = StandardScaler()
numeric_features = pd.DataFrame(standardScaler.fit_transform(numeric_features),
                                columns=numeric_features.columns,
                                index=numeric_features.index)
numeric_features.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>height_cm</th>
      <th>weight_kg</th>
      <th>body fat_%</th>
      <th>diastolic</th>
      <th>systolic</th>
      <th>gripForce</th>
      <th>sit and bend forward_cm</th>
      <th>sit-ups counts</th>
      <th>broad jump_cm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.339300e+04</td>
      <td>1.339300e+04</td>
      <td>1.339300e+04</td>
      <td>1.339300e+04</td>
      <td>1.339300e+04</td>
      <td>1.339300e+04</td>
      <td>1.339300e+04</td>
      <td>1.339300e+04</td>
      <td>1.339300e+04</td>
      <td>1.339300e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.647712e-16</td>
      <td>-8.891649e-16</td>
      <td>5.046228e-16</td>
      <td>-1.134180e-16</td>
      <td>1.522650e-15</td>
      <td>8.187449e-16</td>
      <td>-3.341736e-17</td>
      <td>-4.555122e-17</td>
      <td>6.244538e-17</td>
      <td>3.375101e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000037e+00</td>
      <td>1.000037e+00</td>
      <td>1.000037e+00</td>
      <td>1.000037e+00</td>
      <td>1.000037e+00</td>
      <td>1.000037e+00</td>
      <td>1.000037e+00</td>
      <td>1.000037e+00</td>
      <td>1.000037e+00</td>
      <td>1.000037e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.157795e+00</td>
      <td>-5.169526e+00</td>
      <td>-3.443515e+00</td>
      <td>-2.789218e+00</td>
      <td>-7.335649e+00</td>
      <td>-8.851440e+00</td>
      <td>-3.479128e+00</td>
      <td>-4.754914e+00</td>
      <td>-2.785848e+00</td>
      <td>-4.769156e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-8.642197e-01</td>
      <td>-7.310244e-01</td>
      <td>-7.738845e-01</td>
      <td>-7.221267e-01</td>
      <td>-7.258526e-01</td>
      <td>-6.956117e-01</td>
      <td>-8.907625e-01</td>
      <td>-5.095890e-01</td>
      <td>-6.844432e-01</td>
      <td>-7.055954e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-3.504632e-01</td>
      <td>7.597582e-02</td>
      <td>-3.959736e-03</td>
      <td>-6.065741e-02</td>
      <td>1.891317e-02</td>
      <td>-1.595937e-02</td>
      <td>8.811007e-02</td>
      <td>1.171582e-01</td>
      <td>8.607187e-02</td>
      <td>7.199959e-02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.238375e-01</td>
      <td>7.405642e-01</td>
      <td>6.571713e-01</td>
      <td>6.559343e-01</td>
      <td>6.705832e-01</td>
      <td>7.316582e-01</td>
      <td>7.752033e-01</td>
      <td>6.493019e-01</td>
      <td>7.164933e-01</td>
      <td>7.743435e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.998138e+00</td>
      <td>2.995418e+00</td>
      <td>5.912744e+00</td>
      <td>7.601361e+00</td>
      <td>7.205903e+00</td>
      <td>4.809572e+00</td>
      <td>3.156499e+00</td>
      <td>2.338958e+01</td>
      <td>2.817898e+00</td>
      <td>2.831208e+00</td>
    </tr>
  </tbody>
</table>
</div>



The gender is converted from F and M to 0 and 1.


```python
categorical_features = features[['gender']].copy()
categorical_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>




```python
gender_dict = {'F':0, 'M':1}
categorical_features['gender'].replace(gender_dict, inplace = True)
categorical_features.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4144</th>
      <td>1</td>
    </tr>
    <tr>
      <th>9704</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3434</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2760</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4371</th>
      <td>0</td>
    </tr>
    <tr>
      <th>7887</th>
      <td>0</td>
    </tr>
    <tr>
      <th>6894</th>
      <td>0</td>
    </tr>
    <tr>
      <th>13269</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4974</th>
      <td>0</td>
    </tr>
    <tr>
      <th>9189</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The target variable, class is converted into dummy variables. This is a way of quantifying categorical variables by splitting them into one number for each category (0 and 1).


```python
target = pd.get_dummies(target)
target.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class_A</th>
      <th>class_B</th>
      <th>class_C</th>
      <th>class_D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The categorical and numerical features are joined.


```python
processed_features = pd.concat([categorical_features, numeric_features], axis = 1, sort = False)
processed_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>height_cm</th>
      <th>weight_kg</th>
      <th>body fat_%</th>
      <th>diastolic</th>
      <th>systolic</th>
      <th>gripForce</th>
      <th>sit and bend forward_cm</th>
      <th>sit-ups counts</th>
      <th>broad jump_cm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-0.717432</td>
      <td>0.443873</td>
      <td>0.652150</td>
      <td>-0.267367</td>
      <td>0.112009</td>
      <td>-0.015959</td>
      <td>1.688190</td>
      <td>0.377317</td>
      <td>1.416961</td>
      <td>0.674009</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.864220</td>
      <td>-0.422465</td>
      <td>-0.974734</td>
      <td>-1.039081</td>
      <td>-0.167278</td>
      <td>-0.287820</td>
      <td>-0.053073</td>
      <td>0.128984</td>
      <td>0.926634</td>
      <td>0.975013</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-0.423857</td>
      <td>1.310211</td>
      <td>0.883127</td>
      <td>-0.432734</td>
      <td>1.229158</td>
      <td>1.479276</td>
      <td>0.737554</td>
      <td>-0.379509</td>
      <td>0.646446</td>
      <td>-0.229005</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-0.350463</td>
      <td>0.704961</td>
      <td>0.305684</td>
      <td>-0.667004</td>
      <td>-0.260374</td>
      <td>1.139450</td>
      <td>0.417538</td>
      <td>-0.001096</td>
      <td>0.926634</td>
      <td>0.724176</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>-0.644038</td>
      <td>0.621888</td>
      <td>0.021147</td>
      <td>-0.846152</td>
      <td>-0.818948</td>
      <td>-0.219855</td>
      <td>0.615195</td>
      <td>1.406129</td>
      <td>0.366259</td>
      <td>0.674009</td>
    </tr>
  </tbody>
</table>
</div>



The data is split between training and testing data, in order to avoid overfitting.


```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(processed_features,
                                                   target,
                                                   test_size = 0.2,
                                                   random_state=1)
```

## Creating the Model

The model is made up of 2 layers: one hidden layer that contains 22 neurons, twice as many as there are features in the dataset. This avoids overfitting and underfitting by having neither too few nor too many neurons inside it. 

The second layer has softmax activation that converts vectors into probabilities. This is because it is predicting categorical data.

Adam is the optimising algorithm with a 0.001 learning rate. A low learning rate is used since each predictor follows distribution curves as seen on the Kaggle page above.

Categorical crossentropy is used as it is ideal for multiclasss classification


```python
def build_model():
    model = tf.keras.Sequential([
        layers.Dense(2 * len(x_train.keys()), activation ='relu', input_shape = [len(x_train.keys())]),
        layers.Dense(4, activation = 'softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(0.001)
    
    model.compile(loss='categorical_crossentropy',
                 optimizer="adam",
                 metrics=['accuracy'])
    return model
```


```python
model = build_model()
model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_8 (Dense)             (None, 22)                264       
                                                                     
     dense_9 (Dense)             (None, 4)                 92        
                                                                     
    =================================================================
    Total params: 356
    Trainable params: 356
    Non-trainable params: 0
    _________________________________________________________________


The model is trained here, with early stopping at a patience of 10, to avoid overfitting by training the model on the training data so much that the loss hardly changes. It also has a maximum of 200 epochs to minimise overfitting.


```python
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
training_hist = model.fit(x_train,
                                 y_train,
                                 epochs = 200,
                                 validation_split = 0.2,
                                 verbose = True,
                                 callbacks=[early_stop])
```

    Epoch 1/200
    268/268 [==============================] - 1s 2ms/step - loss: 1.2276 - accuracy: 0.4486 - val_loss: 1.0543 - val_accuracy: 0.5446
    Epoch 2/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.9914 - accuracy: 0.5606 - val_loss: 0.9360 - val_accuracy: 0.5712
    Epoch 3/200
    268/268 [==============================] - 0s 1ms/step - loss: 0.9198 - accuracy: 0.5880 - val_loss: 0.8911 - val_accuracy: 0.5992
    Epoch 4/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.8879 - accuracy: 0.6018 - val_loss: 0.8724 - val_accuracy: 0.6006
    Epoch 5/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.8708 - accuracy: 0.6123 - val_loss: 0.8584 - val_accuracy: 0.6113
    Epoch 6/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.8587 - accuracy: 0.6238 - val_loss: 0.8496 - val_accuracy: 0.6113
    Epoch 7/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.8497 - accuracy: 0.6240 - val_loss: 0.8416 - val_accuracy: 0.6290
    Epoch 8/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.8422 - accuracy: 0.6314 - val_loss: 0.8314 - val_accuracy: 0.6337
    Epoch 9/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.8340 - accuracy: 0.6360 - val_loss: 0.8279 - val_accuracy: 0.6295
    Epoch 10/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.8267 - accuracy: 0.6388 - val_loss: 0.8201 - val_accuracy: 0.6384
    Epoch 11/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.8192 - accuracy: 0.6461 - val_loss: 0.8144 - val_accuracy: 0.6458
    Epoch 12/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.8123 - accuracy: 0.6464 - val_loss: 0.8057 - val_accuracy: 0.6496
    Epoch 13/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.8041 - accuracy: 0.6528 - val_loss: 0.7977 - val_accuracy: 0.6514
    Epoch 14/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7971 - accuracy: 0.6597 - val_loss: 0.7980 - val_accuracy: 0.6584
    Epoch 15/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7902 - accuracy: 0.6630 - val_loss: 0.7869 - val_accuracy: 0.6584
    Epoch 16/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7837 - accuracy: 0.6630 - val_loss: 0.7817 - val_accuracy: 0.6668
    Epoch 17/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7776 - accuracy: 0.6687 - val_loss: 0.7753 - val_accuracy: 0.6678
    Epoch 18/200
    268/268 [==============================] - 0s 1ms/step - loss: 0.7716 - accuracy: 0.6705 - val_loss: 0.7723 - val_accuracy: 0.6692
    Epoch 19/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7677 - accuracy: 0.6754 - val_loss: 0.7666 - val_accuracy: 0.6743
    Epoch 20/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7630 - accuracy: 0.6781 - val_loss: 0.7631 - val_accuracy: 0.6766
    Epoch 21/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7580 - accuracy: 0.6796 - val_loss: 0.7612 - val_accuracy: 0.6785
    Epoch 22/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7548 - accuracy: 0.6773 - val_loss: 0.7557 - val_accuracy: 0.6836
    Epoch 23/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7505 - accuracy: 0.6824 - val_loss: 0.7526 - val_accuracy: 0.6822
    Epoch 24/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7461 - accuracy: 0.6864 - val_loss: 0.7500 - val_accuracy: 0.6804
    Epoch 25/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7426 - accuracy: 0.6855 - val_loss: 0.7481 - val_accuracy: 0.6892
    Epoch 26/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7392 - accuracy: 0.6898 - val_loss: 0.7456 - val_accuracy: 0.6925
    Epoch 27/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7356 - accuracy: 0.6887 - val_loss: 0.7432 - val_accuracy: 0.6850
    Epoch 28/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7315 - accuracy: 0.6935 - val_loss: 0.7397 - val_accuracy: 0.6878
    Epoch 29/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7291 - accuracy: 0.6948 - val_loss: 0.7336 - val_accuracy: 0.6902
    Epoch 30/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7253 - accuracy: 0.6962 - val_loss: 0.7333 - val_accuracy: 0.6888
    Epoch 31/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7224 - accuracy: 0.6979 - val_loss: 0.7294 - val_accuracy: 0.6944
    Epoch 32/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7196 - accuracy: 0.6990 - val_loss: 0.7275 - val_accuracy: 0.6930
    Epoch 33/200
    268/268 [==============================] - 1s 2ms/step - loss: 0.7160 - accuracy: 0.7003 - val_loss: 0.7264 - val_accuracy: 0.6967
    Epoch 34/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7133 - accuracy: 0.7024 - val_loss: 0.7234 - val_accuracy: 0.6916
    Epoch 35/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7105 - accuracy: 0.7039 - val_loss: 0.7188 - val_accuracy: 0.6930
    Epoch 36/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7076 - accuracy: 0.7062 - val_loss: 0.7179 - val_accuracy: 0.6911
    Epoch 37/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7047 - accuracy: 0.7081 - val_loss: 0.7154 - val_accuracy: 0.6972
    Epoch 38/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.7026 - accuracy: 0.7076 - val_loss: 0.7153 - val_accuracy: 0.6976
    Epoch 39/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6999 - accuracy: 0.7107 - val_loss: 0.7120 - val_accuracy: 0.6962
    Epoch 40/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6978 - accuracy: 0.7110 - val_loss: 0.7096 - val_accuracy: 0.7000
    Epoch 41/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6948 - accuracy: 0.7156 - val_loss: 0.7054 - val_accuracy: 0.6958
    Epoch 42/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6927 - accuracy: 0.7129 - val_loss: 0.7039 - val_accuracy: 0.6981
    Epoch 43/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6906 - accuracy: 0.7180 - val_loss: 0.7034 - val_accuracy: 0.6939
    Epoch 44/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6887 - accuracy: 0.7186 - val_loss: 0.7007 - val_accuracy: 0.6972
    Epoch 45/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6863 - accuracy: 0.7167 - val_loss: 0.6990 - val_accuracy: 0.6962
    Epoch 46/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6845 - accuracy: 0.7170 - val_loss: 0.6987 - val_accuracy: 0.6981
    Epoch 47/200
    268/268 [==============================] - 0s 1ms/step - loss: 0.6821 - accuracy: 0.7213 - val_loss: 0.6991 - val_accuracy: 0.6911
    Epoch 48/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6806 - accuracy: 0.7209 - val_loss: 0.6953 - val_accuracy: 0.6948
    Epoch 49/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6793 - accuracy: 0.7249 - val_loss: 0.6927 - val_accuracy: 0.6944
    Epoch 50/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6769 - accuracy: 0.7226 - val_loss: 0.6920 - val_accuracy: 0.7009
    Epoch 51/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6752 - accuracy: 0.7217 - val_loss: 0.6909 - val_accuracy: 0.7000
    Epoch 52/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6739 - accuracy: 0.7252 - val_loss: 0.6904 - val_accuracy: 0.6972
    Epoch 53/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6725 - accuracy: 0.7241 - val_loss: 0.6874 - val_accuracy: 0.6995
    Epoch 54/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6713 - accuracy: 0.7254 - val_loss: 0.6865 - val_accuracy: 0.6986
    Epoch 55/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6696 - accuracy: 0.7250 - val_loss: 0.6874 - val_accuracy: 0.7074
    Epoch 56/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6680 - accuracy: 0.7286 - val_loss: 0.6846 - val_accuracy: 0.7004
    Epoch 57/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6663 - accuracy: 0.7269 - val_loss: 0.6861 - val_accuracy: 0.7014
    Epoch 58/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6653 - accuracy: 0.7279 - val_loss: 0.6834 - val_accuracy: 0.7037
    Epoch 59/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6640 - accuracy: 0.7287 - val_loss: 0.6866 - val_accuracy: 0.6990
    Epoch 60/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6623 - accuracy: 0.7299 - val_loss: 0.6818 - val_accuracy: 0.7042
    Epoch 61/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6612 - accuracy: 0.7310 - val_loss: 0.6867 - val_accuracy: 0.7051
    Epoch 62/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6605 - accuracy: 0.7325 - val_loss: 0.6821 - val_accuracy: 0.7051
    Epoch 63/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6587 - accuracy: 0.7328 - val_loss: 0.6796 - val_accuracy: 0.7065
    Epoch 64/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6583 - accuracy: 0.7339 - val_loss: 0.6780 - val_accuracy: 0.7032
    Epoch 65/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6579 - accuracy: 0.7315 - val_loss: 0.6800 - val_accuracy: 0.6981
    Epoch 66/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6562 - accuracy: 0.7340 - val_loss: 0.6768 - val_accuracy: 0.7065
    Epoch 67/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6548 - accuracy: 0.7334 - val_loss: 0.6800 - val_accuracy: 0.7074
    Epoch 68/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6552 - accuracy: 0.7341 - val_loss: 0.6767 - val_accuracy: 0.6995
    Epoch 69/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6534 - accuracy: 0.7377 - val_loss: 0.6745 - val_accuracy: 0.7098
    Epoch 70/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6522 - accuracy: 0.7381 - val_loss: 0.6748 - val_accuracy: 0.7126
    Epoch 71/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6525 - accuracy: 0.7381 - val_loss: 0.6737 - val_accuracy: 0.7093
    Epoch 72/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6511 - accuracy: 0.7385 - val_loss: 0.6709 - val_accuracy: 0.7084
    Epoch 73/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6500 - accuracy: 0.7384 - val_loss: 0.6727 - val_accuracy: 0.7060
    Epoch 74/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6497 - accuracy: 0.7380 - val_loss: 0.6707 - val_accuracy: 0.7121
    Epoch 75/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6484 - accuracy: 0.7418 - val_loss: 0.6720 - val_accuracy: 0.7070
    Epoch 76/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6482 - accuracy: 0.7370 - val_loss: 0.6710 - val_accuracy: 0.7098
    Epoch 77/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6473 - accuracy: 0.7390 - val_loss: 0.6700 - val_accuracy: 0.7116
    Epoch 78/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6468 - accuracy: 0.7395 - val_loss: 0.6690 - val_accuracy: 0.7121
    Epoch 79/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6459 - accuracy: 0.7382 - val_loss: 0.6666 - val_accuracy: 0.7102
    Epoch 80/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6452 - accuracy: 0.7394 - val_loss: 0.6667 - val_accuracy: 0.7163
    Epoch 81/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6449 - accuracy: 0.7401 - val_loss: 0.6685 - val_accuracy: 0.7098
    Epoch 82/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6444 - accuracy: 0.7419 - val_loss: 0.6655 - val_accuracy: 0.7088
    Epoch 83/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6431 - accuracy: 0.7425 - val_loss: 0.6669 - val_accuracy: 0.7172
    Epoch 84/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6431 - accuracy: 0.7415 - val_loss: 0.6653 - val_accuracy: 0.7135
    Epoch 85/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6420 - accuracy: 0.7413 - val_loss: 0.6627 - val_accuracy: 0.7154
    Epoch 86/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6412 - accuracy: 0.7418 - val_loss: 0.6657 - val_accuracy: 0.7191
    Epoch 87/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6403 - accuracy: 0.7450 - val_loss: 0.6639 - val_accuracy: 0.7177
    Epoch 88/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6404 - accuracy: 0.7396 - val_loss: 0.6643 - val_accuracy: 0.7144
    Epoch 89/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6389 - accuracy: 0.7450 - val_loss: 0.6600 - val_accuracy: 0.7149
    Epoch 90/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6387 - accuracy: 0.7436 - val_loss: 0.6609 - val_accuracy: 0.7196
    Epoch 91/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6375 - accuracy: 0.7451 - val_loss: 0.6628 - val_accuracy: 0.7172
    Epoch 92/200
    268/268 [==============================] - 0s 1ms/step - loss: 0.6375 - accuracy: 0.7453 - val_loss: 0.6619 - val_accuracy: 0.7224
    Epoch 93/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6368 - accuracy: 0.7453 - val_loss: 0.6592 - val_accuracy: 0.7177
    Epoch 94/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6365 - accuracy: 0.7439 - val_loss: 0.6600 - val_accuracy: 0.7214
    Epoch 95/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6353 - accuracy: 0.7460 - val_loss: 0.6607 - val_accuracy: 0.7200
    Epoch 96/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6350 - accuracy: 0.7431 - val_loss: 0.6597 - val_accuracy: 0.7214
    Epoch 97/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6344 - accuracy: 0.7432 - val_loss: 0.6593 - val_accuracy: 0.7154
    Epoch 98/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6340 - accuracy: 0.7488 - val_loss: 0.6577 - val_accuracy: 0.7191
    Epoch 99/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6334 - accuracy: 0.7426 - val_loss: 0.6583 - val_accuracy: 0.7191
    Epoch 100/200
    268/268 [==============================] - 0s 1ms/step - loss: 0.6324 - accuracy: 0.7444 - val_loss: 0.6555 - val_accuracy: 0.7205
    Epoch 101/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6322 - accuracy: 0.7464 - val_loss: 0.6577 - val_accuracy: 0.7214
    Epoch 102/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6313 - accuracy: 0.7443 - val_loss: 0.6585 - val_accuracy: 0.7200
    Epoch 103/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6313 - accuracy: 0.7450 - val_loss: 0.6558 - val_accuracy: 0.7210
    Epoch 104/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6304 - accuracy: 0.7464 - val_loss: 0.6524 - val_accuracy: 0.7224
    Epoch 105/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6302 - accuracy: 0.7426 - val_loss: 0.6577 - val_accuracy: 0.7182
    Epoch 106/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6296 - accuracy: 0.7465 - val_loss: 0.6547 - val_accuracy: 0.7210
    Epoch 107/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6286 - accuracy: 0.7429 - val_loss: 0.6562 - val_accuracy: 0.7242
    Epoch 108/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6285 - accuracy: 0.7445 - val_loss: 0.6540 - val_accuracy: 0.7172
    Epoch 109/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6280 - accuracy: 0.7453 - val_loss: 0.6541 - val_accuracy: 0.7233
    Epoch 110/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6274 - accuracy: 0.7444 - val_loss: 0.6538 - val_accuracy: 0.7233
    Epoch 111/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6272 - accuracy: 0.7462 - val_loss: 0.6530 - val_accuracy: 0.7177
    Epoch 112/200
    268/268 [==============================] - 0s 1ms/step - loss: 0.6268 - accuracy: 0.7480 - val_loss: 0.6556 - val_accuracy: 0.7200
    Epoch 113/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6262 - accuracy: 0.7467 - val_loss: 0.6541 - val_accuracy: 0.7224
    Epoch 114/200
    268/268 [==============================] - 0s 2ms/step - loss: 0.6259 - accuracy: 0.7460 - val_loss: 0.6532 - val_accuracy: 0.7196


## Evaluation

The model, having been trained, reaches what seems to be an upper limit found by the early stopping at just below 75% accuracy. Other times when it is trained at a different random state, it reaches a similar accuracy.


```python
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plt.figure(figsize=(10,8))
plotter.plot({'Early Stopping': training_hist}, metric = "accuracy")
plt.ylabel("Accuracy")
plt.show()
```


![png](output_32_0.png)


The model, whenever it does mispredict the class, rarely mispredicts by more than 1 class. It most often predicts too high rather than too low.


```python
y_pred = model.predict(x_test)
y_pred = y_pred > 0.5
```


```python
confusion_matrix = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1))
cm_table = pd.DataFrame(confusion_matrix,
                     index = ['A','B','C', 'D'], 
                     columns = ['A','B','C', 'D'])
```


```python
cm_table.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>606</td>
      <td>35</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>B</th>
      <td>270</td>
      <td>288</td>
      <td>98</td>
      <td>10</td>
    </tr>
    <tr>
      <th>C</th>
      <td>135</td>
      <td>80</td>
      <td>417</td>
      <td>47</td>
    </tr>
    <tr>
      <th>D</th>
      <td>35</td>
      <td>22</td>
      <td>70</td>
      <td>562</td>
    </tr>
  </tbody>
</table>
</div>
