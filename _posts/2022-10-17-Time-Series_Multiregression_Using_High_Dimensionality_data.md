# Time-Series Multiregression Using High-Dimensionality Data
```python
import os
from sklearn.metrics import mean_absolute_error
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
```

# The Dataset

This is a dataset by Kaggle users Aishwarya, Vaishnavi V and Raja CSP. It was made during an Internship at Tactlabs. It can be found [here](https://www.kaggle.com/datasets/aishu200023/stackindex). It was released under the [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode) license.

It contains data regarding how many questions were asked about Python and Python libraries each month from January 2009 to December 2019. There are 80 different libraries in the dataset. I will be using a time series of these questions to make a prediction on how many questions will be asked regarding Python in the next month.

# Pre-processing

First, the data is loaded and the only non-numerical column (the month) is quantified.


```python
df = pd.read_csv("MLTollsStackOverflow.csv")
```


```python
df.head()
```




<div style="overflow-x:auto;">
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
      <th>month</th>
      <th>nltk</th>
      <th>spacy</th>
      <th>stanford-nlp</th>
      <th>python</th>
      <th>r</th>
      <th>numpy</th>
      <th>scipy</th>
      <th>matlab</th>
      <th>machine-learning</th>
      <th>pandas</th>
      <th>pytorch</th>
      <th>keras</th>
      <th>nlp</th>
      <th>apache-spark</th>
      <th>hadoop</th>
      <th>pyspark</th>
      <th>python-3.x</th>
      <th>tensorflow</th>
      <th>deep-learning</th>
      <th>neural-network</th>
      <th>lstm</th>
      <th>time-series</th>
      <th>pillow</th>
      <th>rasa</th>
      <th>opencv</th>
      <th>pipenv</th>
      <th>seaborn</th>
      <th>Dask</th>
      <th>jupyter</th>
      <th>AllenNLP</th>
      <th>Theano</th>
      <th>plotly</th>
      <th>scikit-learn</th>
      <th>BeautifulSoup</th>
      <th>scrapy</th>
      <th>Gensim</th>
      <th>FastText</th>
      <th>Pydot</th>
      <th>Pybrain</th>
      <th>Pytil</th>
      <th>Pygame</th>
      <th>Colab</th>
      <th>Shogun</th>
      <th>KNIME</th>
      <th>Apache</th>
      <th>Gunicorn</th>
      <th>Pygtk</th>
      <th>Weka</th>
      <th>Conda</th>
      <th>Ray</th>
      <th>matlab.1</th>
      <th>accord.net</th>
      <th>regression</th>
      <th>classification</th>
      <th>correlation</th>
      <th>cluster-analysis</th>
      <th>H2o</th>
      <th>Mallet</th>
      <th>Numba</th>
      <th>Tableau</th>
      <th>Trifacta</th>
      <th>PyArrow</th>
      <th>Rasterio</th>
      <th>Orange3</th>
      <th>PyMC3</th>
      <th>Opennn</th>
      <th>Oryx</th>
      <th>Istio</th>
      <th>Venes</th>
      <th>Plotnine</th>
      <th>Gluon</th>
      <th>Plato</th>
      <th>Sympy</th>
      <th>Flair</th>
      <th>stanford-nlp.1</th>
      <th>pyqt</th>
      <th>Nolearn</th>
      <th>Lasagne</th>
      <th>OCR</th>
      <th>Apache-spark-mlib</th>
      <th>azure-virtual-machine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>09-Jan</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>631</td>
      <td>8</td>
      <td>6</td>
      <td>2</td>
      <td>19</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>95</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>09-Feb</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>633</td>
      <td>9</td>
      <td>7</td>
      <td>3</td>
      <td>27</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>114</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>27</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>09-Mar</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>766</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>24</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>104</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>09-Apr</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>768</td>
      <td>12</td>
      <td>6</td>
      <td>3</td>
      <td>32</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>111</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>09-May</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1003</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>42</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>127</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["month"] = np.arange(len(df))
```


```python
df.head()
```




<div style="overflow-x:auto;">
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
      <th>month</th>
      <th>nltk</th>
      <th>spacy</th>
      <th>stanford-nlp</th>
      <th>python</th>
      <th>r</th>
      <th>numpy</th>
      <th>scipy</th>
      <th>matlab</th>
      <th>machine-learning</th>
      <th>pandas</th>
      <th>pytorch</th>
      <th>keras</th>
      <th>nlp</th>
      <th>apache-spark</th>
      <th>hadoop</th>
      <th>pyspark</th>
      <th>python-3.x</th>
      <th>tensorflow</th>
      <th>deep-learning</th>
      <th>neural-network</th>
      <th>lstm</th>
      <th>time-series</th>
      <th>pillow</th>
      <th>rasa</th>
      <th>opencv</th>
      <th>pipenv</th>
      <th>seaborn</th>
      <th>Dask</th>
      <th>jupyter</th>
      <th>AllenNLP</th>
      <th>Theano</th>
      <th>plotly</th>
      <th>scikit-learn</th>
      <th>BeautifulSoup</th>
      <th>scrapy</th>
      <th>Gensim</th>
      <th>FastText</th>
      <th>Pydot</th>
      <th>Pybrain</th>
      <th>Pytil</th>
      <th>Pygame</th>
      <th>Colab</th>
      <th>Shogun</th>
      <th>KNIME</th>
      <th>Apache</th>
      <th>Gunicorn</th>
      <th>Pygtk</th>
      <th>Weka</th>
      <th>Conda</th>
      <th>Ray</th>
      <th>matlab.1</th>
      <th>accord.net</th>
      <th>regression</th>
      <th>classification</th>
      <th>correlation</th>
      <th>cluster-analysis</th>
      <th>H2o</th>
      <th>Mallet</th>
      <th>Numba</th>
      <th>Tableau</th>
      <th>Trifacta</th>
      <th>PyArrow</th>
      <th>Rasterio</th>
      <th>Orange3</th>
      <th>PyMC3</th>
      <th>Opennn</th>
      <th>Oryx</th>
      <th>Istio</th>
      <th>Venes</th>
      <th>Plotnine</th>
      <th>Gluon</th>
      <th>Plato</th>
      <th>Sympy</th>
      <th>Flair</th>
      <th>stanford-nlp.1</th>
      <th>pyqt</th>
      <th>Nolearn</th>
      <th>Lasagne</th>
      <th>OCR</th>
      <th>Apache-spark-mlib</th>
      <th>azure-virtual-machine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>631</td>
      <td>8</td>
      <td>6</td>
      <td>2</td>
      <td>19</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>95</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>633</td>
      <td>9</td>
      <td>7</td>
      <td>3</td>
      <td>27</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>114</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>27</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>766</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>24</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>104</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>768</td>
      <td>12</td>
      <td>6</td>
      <td>3</td>
      <td>32</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>111</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1003</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>42</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>127</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (132, 82)



## Dimensionality Reduction

There are 82 columns and 132 rows. This is concerningly high dimensionality for a time-series dataset so first I will need to considerably reduce that.
### Removing Empty or Redundant Features
It is quite visible here that some columns are over 25% empty. Tableau is the only column containing missing data as its count is below 132. I will start by removing these two. These are mostly libraries or concepts that were not introduced until after 2009, making that data not representative of the next year's trends. That would undermine the model's ability to predict future python question count.

Python's coefficient of variation is 0.6311512408227511. It has a very slight positive skew of 0.17354069870622177. It has quite a low kurtosis value of -1.1515067392234757.


```python
df.count()
```




    month                    132
    nltk                     132
    spacy                    132
    stanford-nlp             132
    python                   132
    r                        132
    numpy                    132
    scipy                    132
    matlab                   132
    machine-learning         132
    pandas                   132
    pytorch                  132
    keras                    132
    nlp                      132
    apache-spark             132
    hadoop                   132
    pyspark                  132
    python-3.x               132
    tensorflow               132
    deep-learning            132
    neural-network           132
    lstm                     132
    time-series              132
    pillow                   132
    rasa                     132
    opencv                   132
    pipenv                   132
    seaborn                  132
    Dask                     132
    jupyter                  132
    AllenNLP                 132
    Theano                   132
    plotly                   132
    scikit-learn             132
    BeautifulSoup            132
    scrapy                   132
    Gensim                   132
    FastText                 132
    Pydot                    132
    Pybrain                  132
    Pytil                    132
    Pygame                   132
    Colab                    132
    Shogun                   132
    KNIME                    132
    Apache                   132
    Gunicorn                 132
    Pygtk                    132
    Weka                     132
    Conda                    132
    Ray                      132
    matlab.1                 132
    accord.net               132
    regression               132
    classification           132
    correlation              132
    cluster-analysis         132
    H2o                      132
    Mallet                   132
    Numba                    132
    Tableau                  108
    Trifacta                 132
    PyArrow                  132
    Rasterio                 132
    Orange3                  132
    PyMC3                    132
    Opennn                   132
    Oryx                     132
    Istio                    132
    Venes                    132
    Plotnine                 132
    Gluon                    132
    Plato                    132
    Sympy                    132
    Flair                    132
    stanford-nlp.1           132
    pyqt                     132
    Nolearn                  132
    Lasagne                  132
    OCR                      132
    Apache-spark-mlib        132
    azure-virtual-machine    132
    dtype: int64




```python
pd.set_option('display.max_columns', 500)
df.describe()
```




<div style="overflow-x:auto;">
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
      <th>month</th>
      <th>nltk</th>
      <th>spacy</th>
      <th>stanford-nlp</th>
      <th>python</th>
      <th>r</th>
      <th>numpy</th>
      <th>scipy</th>
      <th>matlab</th>
      <th>machine-learning</th>
      <th>pandas</th>
      <th>pytorch</th>
      <th>keras</th>
      <th>nlp</th>
      <th>apache-spark</th>
      <th>hadoop</th>
      <th>pyspark</th>
      <th>python-3.x</th>
      <th>tensorflow</th>
      <th>deep-learning</th>
      <th>neural-network</th>
      <th>lstm</th>
      <th>time-series</th>
      <th>pillow</th>
      <th>rasa</th>
      <th>opencv</th>
      <th>pipenv</th>
      <th>seaborn</th>
      <th>Dask</th>
      <th>jupyter</th>
      <th>AllenNLP</th>
      <th>Theano</th>
      <th>plotly</th>
      <th>scikit-learn</th>
      <th>BeautifulSoup</th>
      <th>scrapy</th>
      <th>Gensim</th>
      <th>FastText</th>
      <th>Pydot</th>
      <th>Pybrain</th>
      <th>Pytil</th>
      <th>Pygame</th>
      <th>Colab</th>
      <th>Shogun</th>
      <th>KNIME</th>
      <th>Apache</th>
      <th>Gunicorn</th>
      <th>Pygtk</th>
      <th>Weka</th>
      <th>Conda</th>
      <th>Ray</th>
      <th>matlab.1</th>
      <th>accord.net</th>
      <th>regression</th>
      <th>classification</th>
      <th>correlation</th>
      <th>cluster-analysis</th>
      <th>H2o</th>
      <th>Mallet</th>
      <th>Numba</th>
      <th>Tableau</th>
      <th>Trifacta</th>
      <th>PyArrow</th>
      <th>Rasterio</th>
      <th>Orange3</th>
      <th>PyMC3</th>
      <th>Opennn</th>
      <th>Oryx</th>
      <th>Istio</th>
      <th>Venes</th>
      <th>Plotnine</th>
      <th>Gluon</th>
      <th>Plato</th>
      <th>Sympy</th>
      <th>Flair</th>
      <th>stanford-nlp.1</th>
      <th>pyqt</th>
      <th>Nolearn</th>
      <th>Lasagne</th>
      <th>OCR</th>
      <th>Apache-spark-mlib</th>
      <th>azure-virtual-machine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.00000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>108.0</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.0</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
      <td>132.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>65.500000</td>
      <td>42.704545</td>
      <td>11.848485</td>
      <td>25.537879</td>
      <td>9856.704545</td>
      <td>2411.856061</td>
      <td>514.204545</td>
      <td>112.454545</td>
      <td>651.681818</td>
      <td>264.401515</td>
      <td>980.295455</td>
      <td>36.401515</td>
      <td>163.924242</td>
      <td>80.787879</td>
      <td>414.833333</td>
      <td>309.143939</td>
      <td>54.022727</td>
      <td>1478.227273</td>
      <td>361.090909</td>
      <td>105.424242</td>
      <td>107.636364</td>
      <td>26.901515</td>
      <td>67.318182</td>
      <td>8.484848</td>
      <td>0.780303</td>
      <td>401.659091</td>
      <td>4.409091</td>
      <td>29.371212</td>
      <td>16.537879</td>
      <td>30.560606</td>
      <td>0.189394</td>
      <td>17.909091</td>
      <td>38.196970</td>
      <td>124.992424</td>
      <td>134.090909</td>
      <td>95.295455</td>
      <td>12.128788</td>
      <td>1.545455</td>
      <td>1.250000</td>
      <td>2.310606</td>
      <td>0.030303</td>
      <td>87.613636</td>
      <td>24.371212</td>
      <td>0.371212</td>
      <td>1.696970</td>
      <td>632.757576</td>
      <td>20.484848</td>
      <td>15.863636</td>
      <td>20.787879</td>
      <td>24.371212</td>
      <td>1.189394</td>
      <td>651.30303</td>
      <td>2.007576</td>
      <td>43.409091</td>
      <td>42.454545</td>
      <td>17.446970</td>
      <td>33.848485</td>
      <td>11.356061</td>
      <td>2.128788</td>
      <td>7.159091</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>2.090909</td>
      <td>1.000000</td>
      <td>2.136364</td>
      <td>3.636364</td>
      <td>0.022727</td>
      <td>0.068182</td>
      <td>6.621212</td>
      <td>0.0</td>
      <td>0.431818</td>
      <td>3.742424</td>
      <td>0.128788</td>
      <td>21.643939</td>
      <td>0.022727</td>
      <td>22.522727</td>
      <td>92.386364</td>
      <td>0.325758</td>
      <td>1.507576</td>
      <td>31.931818</td>
      <td>15.560606</td>
      <td>12.984848</td>
    </tr>
    <tr>
      <th>std</th>
      <td>38.249183</td>
      <td>29.024533</td>
      <td>21.066773</td>
      <td>22.821045</td>
      <td>6221.071304</td>
      <td>1714.759241</td>
      <td>392.882978</td>
      <td>69.680773</td>
      <td>395.949633</td>
      <td>249.655453</td>
      <td>1148.018331</td>
      <td>77.833533</td>
      <td>274.019741</td>
      <td>56.824227</td>
      <td>458.009336</td>
      <td>212.345323</td>
      <td>101.513092</td>
      <td>1820.009209</td>
      <td>522.675698</td>
      <td>145.808571</td>
      <td>100.385917</td>
      <td>40.629185</td>
      <td>50.430561</td>
      <td>10.529845</td>
      <td>3.577526</td>
      <td>218.717770</td>
      <td>9.308089</td>
      <td>37.798260</td>
      <td>25.677508</td>
      <td>38.005274</td>
      <td>0.619438</td>
      <td>25.629317</td>
      <td>50.682282</td>
      <td>131.165154</td>
      <td>110.029452</td>
      <td>70.549721</td>
      <td>14.187084</td>
      <td>3.148642</td>
      <td>1.479504</td>
      <td>2.766022</td>
      <td>0.172073</td>
      <td>59.576066</td>
      <td>34.063570</td>
      <td>0.775634</td>
      <td>2.192769</td>
      <td>289.343192</td>
      <td>15.732732</td>
      <td>11.239769</td>
      <td>14.911459</td>
      <td>34.063570</td>
      <td>3.820660</td>
      <td>396.22921</td>
      <td>2.394320</td>
      <td>34.248297</td>
      <td>26.500285</td>
      <td>11.075945</td>
      <td>18.366699</td>
      <td>16.150595</td>
      <td>2.134433</td>
      <td>8.193117</td>
      <td>0.0</td>
      <td>0.242872</td>
      <td>4.866860</td>
      <td>2.230941</td>
      <td>2.516428</td>
      <td>4.506607</td>
      <td>0.149600</td>
      <td>0.394488</td>
      <td>14.906083</td>
      <td>0.0</td>
      <td>1.113369</td>
      <td>6.155270</td>
      <td>0.378936</td>
      <td>18.749507</td>
      <td>0.194030</td>
      <td>19.393265</td>
      <td>47.025030</td>
      <td>1.037438</td>
      <td>3.185119</td>
      <td>16.307725</td>
      <td>19.942765</td>
      <td>12.420732</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>631.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>19.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>95.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>19.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.750000</td>
      <td>16.750000</td>
      <td>0.000000</td>
      <td>5.750000</td>
      <td>3744.250000</td>
      <td>608.750000</td>
      <td>119.000000</td>
      <td>39.750000</td>
      <td>363.500000</td>
      <td>54.000000</td>
      <td>1.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>42.750000</td>
      <td>0.000000</td>
      <td>104.750000</td>
      <td>0.000000</td>
      <td>95.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>23.250000</td>
      <td>0.000000</td>
      <td>17.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>211.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>28.000000</td>
      <td>17.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>24.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>407.750000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>8.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>358.75000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>16.750000</td>
      <td>8.500000</td>
      <td>18.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>5.750000</td>
      <td>58.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>65.500000</td>
      <td>44.500000</td>
      <td>0.000000</td>
      <td>17.500000</td>
      <td>9651.500000</td>
      <td>2613.500000</td>
      <td>486.000000</td>
      <td>130.500000</td>
      <td>581.000000</td>
      <td>154.500000</td>
      <td>483.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>65.000000</td>
      <td>128.500000</td>
      <td>286.500000</td>
      <td>0.000000</td>
      <td>587.500000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>53.500000</td>
      <td>1.000000</td>
      <td>66.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>485.500000</td>
      <td>0.000000</td>
      <td>5.500000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>5.500000</td>
      <td>2.000000</td>
      <td>90.500000</td>
      <td>112.000000</td>
      <td>110.000000</td>
      <td>4.500000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>101.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>667.000000</td>
      <td>21.000000</td>
      <td>12.000000</td>
      <td>20.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>581.00000</td>
      <td>1.000000</td>
      <td>42.000000</td>
      <td>46.000000</td>
      <td>20.000000</td>
      <td>35.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>98.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
      <td>3.000000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>98.250000</td>
      <td>66.500000</td>
      <td>15.000000</td>
      <td>46.000000</td>
      <td>15590.750000</td>
      <td>4000.500000</td>
      <td>901.500000</td>
      <td>171.250000</td>
      <td>997.750000</td>
      <td>495.250000</td>
      <td>1860.750000</td>
      <td>16.250000</td>
      <td>329.500000</td>
      <td>104.500000</td>
      <td>915.000000</td>
      <td>500.500000</td>
      <td>34.250000</td>
      <td>2426.250000</td>
      <td>970.000000</td>
      <td>234.250000</td>
      <td>194.500000</td>
      <td>59.750000</td>
      <td>105.000000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>552.500000</td>
      <td>0.000000</td>
      <td>50.500000</td>
      <td>31.750000</td>
      <td>66.250000</td>
      <td>0.000000</td>
      <td>21.000000</td>
      <td>82.250000</td>
      <td>247.000000</td>
      <td>233.000000</td>
      <td>152.250000</td>
      <td>27.000000</td>
      <td>1.250000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>129.500000</td>
      <td>42.500000</td>
      <td>0.250000</td>
      <td>3.000000</td>
      <td>883.250000</td>
      <td>31.000000</td>
      <td>23.000000</td>
      <td>31.000000</td>
      <td>42.500000</td>
      <td>0.000000</td>
      <td>997.75000</td>
      <td>4.000000</td>
      <td>70.500000</td>
      <td>62.000000</td>
      <td>26.000000</td>
      <td>47.250000</td>
      <td>20.500000</td>
      <td>3.000000</td>
      <td>12.250000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>38.000000</td>
      <td>0.000000</td>
      <td>38.000000</td>
      <td>124.000000</td>
      <td>0.000000</td>
      <td>1.250000</td>
      <td>44.000000</td>
      <td>28.250000</td>
      <td>21.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>131.000000</td>
      <td>106.000000</td>
      <td>79.000000</td>
      <td>79.000000</td>
      <td>23602.000000</td>
      <td>5138.000000</td>
      <td>1310.000000</td>
      <td>229.000000</td>
      <td>1535.000000</td>
      <td>983.000000</td>
      <td>4098.000000</td>
      <td>343.000000</td>
      <td>929.000000</td>
      <td>272.000000</td>
      <td>1218.000000</td>
      <td>693.000000</td>
      <td>351.000000</td>
      <td>6621.000000</td>
      <td>1513.000000</td>
      <td>500.000000</td>
      <td>343.000000</td>
      <td>137.000000</td>
      <td>203.000000</td>
      <td>32.000000</td>
      <td>25.000000</td>
      <td>807.000000</td>
      <td>40.000000</td>
      <td>133.000000</td>
      <td>91.000000</td>
      <td>132.000000</td>
      <td>4.000000</td>
      <td>89.000000</td>
      <td>174.000000</td>
      <td>439.000000</td>
      <td>376.000000</td>
      <td>219.000000</td>
      <td>49.000000</td>
      <td>15.000000</td>
      <td>6.000000</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>230.000000</td>
      <td>125.000000</td>
      <td>4.000000</td>
      <td>9.000000</td>
      <td>1237.000000</td>
      <td>61.000000</td>
      <td>48.000000</td>
      <td>70.000000</td>
      <td>125.000000</td>
      <td>19.000000</td>
      <td>1535.00000</td>
      <td>9.000000</td>
      <td>137.000000</td>
      <td>103.000000</td>
      <td>47.000000</td>
      <td>67.000000</td>
      <td>60.000000</td>
      <td>10.000000</td>
      <td>31.000000</td>
      <td>0.0</td>
      <td>2.000000</td>
      <td>25.000000</td>
      <td>13.000000</td>
      <td>10.000000</td>
      <td>17.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>65.000000</td>
      <td>0.0</td>
      <td>6.000000</td>
      <td>31.000000</td>
      <td>2.000000</td>
      <td>72.000000</td>
      <td>2.000000</td>
      <td>77.000000</td>
      <td>196.000000</td>
      <td>6.000000</td>
      <td>14.000000</td>
      <td>73.000000</td>
      <td>75.000000</td>
      <td>48.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
6221.071304/9856.704545
```




    0.6311512408227511




```python
df["python"].skew()
```




    0.17354069870622177




```python
df["python"].kurt()
```




    -1.1515067392234757



I then removed all columns that are over 20% zeroes. That has reduced the dimensionality down to a significantly more manageable level and removed those data points that may perhaps belong to libraries not present in the earlier parts of the training data (the library may not have been released in 2009 for example. This also removes Tableau which contained null values.


```python
df = df.loc[:, (df==0).mean() < 0.2]
df.head()
```




<div style="overflow-x:auto;">
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
      <th>month</th>
      <th>nltk</th>
      <th>stanford-nlp</th>
      <th>python</th>
      <th>r</th>
      <th>numpy</th>
      <th>scipy</th>
      <th>matlab</th>
      <th>machine-learning</th>
      <th>nlp</th>
      <th>hadoop</th>
      <th>python-3.x</th>
      <th>neural-network</th>
      <th>time-series</th>
      <th>opencv</th>
      <th>BeautifulSoup</th>
      <th>scrapy</th>
      <th>Pygame</th>
      <th>Apache</th>
      <th>Gunicorn</th>
      <th>Pygtk</th>
      <th>Weka</th>
      <th>matlab.1</th>
      <th>regression</th>
      <th>classification</th>
      <th>correlation</th>
      <th>cluster-analysis</th>
      <th>Sympy</th>
      <th>stanford-nlp.1</th>
      <th>pyqt</th>
      <th>OCR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>631</td>
      <td>8</td>
      <td>6</td>
      <td>2</td>
      <td>19</td>
      <td>8</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>95</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>19</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>633</td>
      <td>9</td>
      <td>7</td>
      <td>3</td>
      <td>27</td>
      <td>4</td>
      <td>9</td>
      <td>1</td>
      <td>12</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>114</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>27</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>766</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>24</td>
      <td>3</td>
      <td>12</td>
      <td>1</td>
      <td>4</td>
      <td>11</td>
      <td>1</td>
      <td>7</td>
      <td>8</td>
      <td>0</td>
      <td>7</td>
      <td>104</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>768</td>
      <td>12</td>
      <td>6</td>
      <td>3</td>
      <td>32</td>
      <td>10</td>
      <td>14</td>
      <td>6</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>111</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>32</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1003</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>42</td>
      <td>7</td>
      <td>9</td>
      <td>3</td>
      <td>19</td>
      <td>7</td>
      <td>0</td>
      <td>10</td>
      <td>14</td>
      <td>0</td>
      <td>2</td>
      <td>127</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>42</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["python"].plot()
```




    <AxesSubplot:>




![png](/images/time-series-data/output_40_0.png)


### Analysis Through Decomposition
It can be seen here, that the trend in Python is a seemingly linear upward trend. It has little seasonality that, since it is only made from 10 years of monthly data, could be passed off as by chance alone.


```python
frame = seasonal_decompose(df["python"], model='multiplicative', period = 12)
```


```python
import matplotlib as plt
plt.rcParams.update({'figure.figsize': (5,5)})
frame.plot().suptitle(fontsize=22)
plt.show()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /tmp/ipykernel_14276/2769107739.py in <cell line: 3>()
          1 import matplotlib as plt
          2 plt.rcParams.update({'figure.figsize': (5,5)})
    ----> 3 frame.plot().suptitle(fontsize=22)
          4 plt.show()


    TypeError: suptitle() missing 1 required positional argument: 't'



![png](/images/time-series-data/output_18_1.png)



```python
plt.rcParams.update({'figure.figsize': (5,5)})
arr = frame.seasonal[0:12]
arr.plot()
```

There is no detectable trend in the amount of residual noise, which shows that the use of the multiplucative decomposition was correct.


```python
plt.rcParams.update({'figure.figsize': (50,5)})
frame.resid.plot()
```




    <AxesSubplot:>




![png](/images/time-series-data/output_21_1.png)


### Analysing Correlations
Some features with unusual correlations with other features are found, but upon closer inspection they don't seem to have obvious issues that would make them negatively impact the model's performance. They all tend to rise, reach a peak and fall. I decided not to drop these as these may be of more use than the more correlated variables, seeing as historical data for Python is available when making predictions too.


```python
plt.rcParams.update({'figure.figsize': (40,40)})
fig = plt.pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(), interpolation='nearest')
fig.colorbar(cax)
ax.set_xticks(range(len(df.columns)))
ax.set_yticks(range(len(df.columns)))
ax.set_xticklabels(['']+df.columns)
ax.set_yticklabels(['']+df.columns)

ax
```




    <AxesSubplot:>




![png](/images/time-series-data/output_23_1.png)



```python
plt.rcParams.update({'figure.figsize': (10,10)})
df["Pygtk"].plot()
```




    <AxesSubplot:>




![png](/images/time-series-data/output_24_1.png)



```python
df["Apache"].plot()
```




    <AxesSubplot:>




![png](/images/time-series-data/output_25_1.png)



```python
df["matlab"].plot()
```




    <AxesSubplot:>




![png](/images/time-series-data/output_26_1.png)


In this closer inspection, the near-linear relation between month and python as shown by the decomposition is apparent. One noticeable problem is detectable here: matlab and matlab.1 are very highly correlated. They could reasonably be duplicates, so matlab.1 is removed.


```python
df.corr()
```




<div style="overflow-x:auto;">
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
      <th>month</th>
      <th>nltk</th>
      <th>stanford-nlp</th>
      <th>python</th>
      <th>r</th>
      <th>numpy</th>
      <th>scipy</th>
      <th>matlab</th>
      <th>machine-learning</th>
      <th>nlp</th>
      <th>hadoop</th>
      <th>python-3.x</th>
      <th>neural-network</th>
      <th>time-series</th>
      <th>opencv</th>
      <th>BeautifulSoup</th>
      <th>scrapy</th>
      <th>Pygame</th>
      <th>Apache</th>
      <th>Gunicorn</th>
      <th>Pygtk</th>
      <th>Weka</th>
      <th>matlab.1</th>
      <th>regression</th>
      <th>classification</th>
      <th>correlation</th>
      <th>cluster-analysis</th>
      <th>Sympy</th>
      <th>stanford-nlp.1</th>
      <th>pyqt</th>
      <th>OCR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>month</th>
      <td>1.000000</td>
      <td>0.894761</td>
      <td>0.722115</td>
      <td>0.987061</td>
      <td>0.966456</td>
      <td>0.976022</td>
      <td>0.931809</td>
      <td>0.406995</td>
      <td>0.921440</td>
      <td>0.805533</td>
      <td>0.543964</td>
      <td>0.881236</td>
      <td>0.881922</td>
      <td>0.958101</td>
      <td>0.823003</td>
      <td>0.964120</td>
      <td>0.938008</td>
      <td>0.897563</td>
      <td>0.337570</td>
      <td>0.934885</td>
      <td>-0.590748</td>
      <td>0.292909</td>
      <td>0.405177</td>
      <td>0.932747</td>
      <td>0.912847</td>
      <td>0.883380</td>
      <td>0.856002</td>
      <td>0.924760</td>
      <td>0.646442</td>
      <td>0.910128</td>
      <td>0.866315</td>
    </tr>
    <tr>
      <th>nltk</th>
      <td>0.894761</td>
      <td>1.000000</td>
      <td>0.803787</td>
      <td>0.898785</td>
      <td>0.940620</td>
      <td>0.913537</td>
      <td>0.923672</td>
      <td>0.563544</td>
      <td>0.834591</td>
      <td>0.703690</td>
      <td>0.711687</td>
      <td>0.680240</td>
      <td>0.859821</td>
      <td>0.875334</td>
      <td>0.825930</td>
      <td>0.823409</td>
      <td>0.899411</td>
      <td>0.819469</td>
      <td>0.493519</td>
      <td>0.851715</td>
      <td>-0.571376</td>
      <td>0.430621</td>
      <td>0.562623</td>
      <td>0.898775</td>
      <td>0.920156</td>
      <td>0.850173</td>
      <td>0.843128</td>
      <td>0.885348</td>
      <td>0.806042</td>
      <td>0.815972</td>
      <td>0.798532</td>
    </tr>
    <tr>
      <th>stanford-nlp</th>
      <td>0.722115</td>
      <td>0.803787</td>
      <td>1.000000</td>
      <td>0.710098</td>
      <td>0.791098</td>
      <td>0.722783</td>
      <td>0.784197</td>
      <td>0.533975</td>
      <td>0.637482</td>
      <td>0.403546</td>
      <td>0.688685</td>
      <td>0.500217</td>
      <td>0.687535</td>
      <td>0.674171</td>
      <td>0.669445</td>
      <td>0.634024</td>
      <td>0.769414</td>
      <td>0.620004</td>
      <td>0.459789</td>
      <td>0.682733</td>
      <td>-0.503671</td>
      <td>0.417443</td>
      <td>0.533660</td>
      <td>0.692117</td>
      <td>0.735404</td>
      <td>0.681540</td>
      <td>0.703878</td>
      <td>0.748211</td>
      <td>0.880015</td>
      <td>0.644288</td>
      <td>0.624739</td>
    </tr>
    <tr>
      <th>python</th>
      <td>0.987061</td>
      <td>0.898785</td>
      <td>0.710098</td>
      <td>1.000000</td>
      <td>0.976875</td>
      <td>0.992754</td>
      <td>0.938667</td>
      <td>0.400591</td>
      <td>0.943394</td>
      <td>0.830827</td>
      <td>0.530420</td>
      <td>0.896530</td>
      <td>0.903962</td>
      <td>0.972968</td>
      <td>0.826019</td>
      <td>0.971518</td>
      <td>0.943253</td>
      <td>0.908709</td>
      <td>0.316988</td>
      <td>0.941787</td>
      <td>-0.617162</td>
      <td>0.277733</td>
      <td>0.398713</td>
      <td>0.949591</td>
      <td>0.926796</td>
      <td>0.887138</td>
      <td>0.863666</td>
      <td>0.942827</td>
      <td>0.646793</td>
      <td>0.896023</td>
      <td>0.876846</td>
    </tr>
    <tr>
      <th>r</th>
      <td>0.966456</td>
      <td>0.940620</td>
      <td>0.791098</td>
      <td>0.976875</td>
      <td>1.000000</td>
      <td>0.978732</td>
      <td>0.962789</td>
      <td>0.511965</td>
      <td>0.904557</td>
      <td>0.773163</td>
      <td>0.657224</td>
      <td>0.808032</td>
      <td>0.893865</td>
      <td>0.946101</td>
      <td>0.858591</td>
      <td>0.925860</td>
      <td>0.963228</td>
      <td>0.886094</td>
      <td>0.424820</td>
      <td>0.919196</td>
      <td>-0.634004</td>
      <td>0.367647</td>
      <td>0.510440</td>
      <td>0.935454</td>
      <td>0.938490</td>
      <td>0.885151</td>
      <td>0.888360</td>
      <td>0.941951</td>
      <td>0.761613</td>
      <td>0.871012</td>
      <td>0.853723</td>
    </tr>
    <tr>
      <th>numpy</th>
      <td>0.976022</td>
      <td>0.913537</td>
      <td>0.722783</td>
      <td>0.992754</td>
      <td>0.978732</td>
      <td>1.000000</td>
      <td>0.939659</td>
      <td>0.386854</td>
      <td>0.951116</td>
      <td>0.815459</td>
      <td>0.529622</td>
      <td>0.886093</td>
      <td>0.928631</td>
      <td>0.965873</td>
      <td>0.802378</td>
      <td>0.964891</td>
      <td>0.941293</td>
      <td>0.891209</td>
      <td>0.300755</td>
      <td>0.927438</td>
      <td>-0.648004</td>
      <td>0.251633</td>
      <td>0.385120</td>
      <td>0.948927</td>
      <td>0.920989</td>
      <td>0.875604</td>
      <td>0.850162</td>
      <td>0.947783</td>
      <td>0.665346</td>
      <td>0.870863</td>
      <td>0.859104</td>
    </tr>
    <tr>
      <th>scipy</th>
      <td>0.931809</td>
      <td>0.923672</td>
      <td>0.784197</td>
      <td>0.938667</td>
      <td>0.962789</td>
      <td>0.939659</td>
      <td>1.000000</td>
      <td>0.612774</td>
      <td>0.834988</td>
      <td>0.724759</td>
      <td>0.727267</td>
      <td>0.724866</td>
      <td>0.834261</td>
      <td>0.896210</td>
      <td>0.900917</td>
      <td>0.868467</td>
      <td>0.943975</td>
      <td>0.890109</td>
      <td>0.537619</td>
      <td>0.887493</td>
      <td>-0.554801</td>
      <td>0.459200</td>
      <td>0.610998</td>
      <td>0.897000</td>
      <td>0.922862</td>
      <td>0.883277</td>
      <td>0.883615</td>
      <td>0.888094</td>
      <td>0.760325</td>
      <td>0.868415</td>
      <td>0.829875</td>
    </tr>
    <tr>
      <th>matlab</th>
      <td>0.406995</td>
      <td>0.563544</td>
      <td>0.533975</td>
      <td>0.400591</td>
      <td>0.511965</td>
      <td>0.386854</td>
      <td>0.612774</td>
      <td>1.000000</td>
      <td>0.175043</td>
      <td>0.302401</td>
      <td>0.939204</td>
      <td>-0.011811</td>
      <td>0.221784</td>
      <td>0.380900</td>
      <td>0.783440</td>
      <td>0.243028</td>
      <td>0.530458</td>
      <td>0.532474</td>
      <td>0.951598</td>
      <td>0.416753</td>
      <td>-0.088636</td>
      <td>0.908043</td>
      <td>0.999949</td>
      <td>0.385995</td>
      <td>0.568297</td>
      <td>0.544960</td>
      <td>0.638835</td>
      <td>0.404499</td>
      <td>0.694404</td>
      <td>0.491690</td>
      <td>0.432927</td>
    </tr>
    <tr>
      <th>machine-learning</th>
      <td>0.921440</td>
      <td>0.834591</td>
      <td>0.637482</td>
      <td>0.943394</td>
      <td>0.904557</td>
      <td>0.951116</td>
      <td>0.834988</td>
      <td>0.175043</td>
      <td>1.000000</td>
      <td>0.784089</td>
      <td>0.335976</td>
      <td>0.917574</td>
      <td>0.963629</td>
      <td>0.931827</td>
      <td>0.665658</td>
      <td>0.948894</td>
      <td>0.853931</td>
      <td>0.818508</td>
      <td>0.096910</td>
      <td>0.879681</td>
      <td>-0.638582</td>
      <td>0.073342</td>
      <td>0.173679</td>
      <td>0.927352</td>
      <td>0.869519</td>
      <td>0.786207</td>
      <td>0.758948</td>
      <td>0.909197</td>
      <td>0.540140</td>
      <td>0.797777</td>
      <td>0.832613</td>
    </tr>
    <tr>
      <th>nlp</th>
      <td>0.805533</td>
      <td>0.703690</td>
      <td>0.403546</td>
      <td>0.830827</td>
      <td>0.773163</td>
      <td>0.815459</td>
      <td>0.724759</td>
      <td>0.302401</td>
      <td>0.784089</td>
      <td>1.000000</td>
      <td>0.379665</td>
      <td>0.767013</td>
      <td>0.727213</td>
      <td>0.822282</td>
      <td>0.677920</td>
      <td>0.808054</td>
      <td>0.771901</td>
      <td>0.784766</td>
      <td>0.212115</td>
      <td>0.751478</td>
      <td>-0.463135</td>
      <td>0.198937</td>
      <td>0.299685</td>
      <td>0.780885</td>
      <td>0.771930</td>
      <td>0.718086</td>
      <td>0.715198</td>
      <td>0.763999</td>
      <td>0.503749</td>
      <td>0.703741</td>
      <td>0.776767</td>
    </tr>
    <tr>
      <th>hadoop</th>
      <td>0.543964</td>
      <td>0.711687</td>
      <td>0.688685</td>
      <td>0.530420</td>
      <td>0.657224</td>
      <td>0.529622</td>
      <td>0.727267</td>
      <td>0.939204</td>
      <td>0.335976</td>
      <td>0.379665</td>
      <td>1.000000</td>
      <td>0.121579</td>
      <td>0.407025</td>
      <td>0.502718</td>
      <td>0.819273</td>
      <td>0.385708</td>
      <td>0.667057</td>
      <td>0.600705</td>
      <td>0.911161</td>
      <td>0.530608</td>
      <td>-0.273881</td>
      <td>0.813893</td>
      <td>0.938936</td>
      <td>0.518338</td>
      <td>0.667566</td>
      <td>0.622974</td>
      <td>0.697325</td>
      <td>0.546557</td>
      <td>0.834700</td>
      <td>0.569706</td>
      <td>0.514376</td>
    </tr>
    <tr>
      <th>python-3.x</th>
      <td>0.881236</td>
      <td>0.680240</td>
      <td>0.500217</td>
      <td>0.896530</td>
      <td>0.808032</td>
      <td>0.886093</td>
      <td>0.724866</td>
      <td>-0.011811</td>
      <td>0.917574</td>
      <td>0.767013</td>
      <td>0.121579</td>
      <td>1.000000</td>
      <td>0.829425</td>
      <td>0.887532</td>
      <td>0.546685</td>
      <td>0.945113</td>
      <td>0.764296</td>
      <td>0.754918</td>
      <td>-0.105769</td>
      <td>0.832875</td>
      <td>-0.620027</td>
      <td>-0.091383</td>
      <td>-0.014205</td>
      <td>0.845012</td>
      <td>0.737107</td>
      <td>0.719696</td>
      <td>0.653759</td>
      <td>0.826870</td>
      <td>0.336663</td>
      <td>0.769922</td>
      <td>0.749499</td>
    </tr>
    <tr>
      <th>neural-network</th>
      <td>0.881922</td>
      <td>0.859821</td>
      <td>0.687535</td>
      <td>0.903962</td>
      <td>0.893865</td>
      <td>0.928631</td>
      <td>0.834261</td>
      <td>0.221784</td>
      <td>0.963629</td>
      <td>0.727213</td>
      <td>0.407025</td>
      <td>0.829425</td>
      <td>1.000000</td>
      <td>0.886994</td>
      <td>0.658434</td>
      <td>0.896149</td>
      <td>0.844406</td>
      <td>0.775641</td>
      <td>0.157582</td>
      <td>0.833882</td>
      <td>-0.627745</td>
      <td>0.104490</td>
      <td>0.220783</td>
      <td>0.888283</td>
      <td>0.863275</td>
      <td>0.751245</td>
      <td>0.733793</td>
      <td>0.893532</td>
      <td>0.624416</td>
      <td>0.738384</td>
      <td>0.806459</td>
    </tr>
    <tr>
      <th>time-series</th>
      <td>0.958101</td>
      <td>0.875334</td>
      <td>0.674171</td>
      <td>0.972968</td>
      <td>0.946101</td>
      <td>0.965873</td>
      <td>0.896210</td>
      <td>0.380900</td>
      <td>0.931827</td>
      <td>0.822282</td>
      <td>0.502718</td>
      <td>0.887532</td>
      <td>0.886994</td>
      <td>1.000000</td>
      <td>0.805804</td>
      <td>0.943511</td>
      <td>0.902404</td>
      <td>0.896524</td>
      <td>0.286173</td>
      <td>0.926887</td>
      <td>-0.611334</td>
      <td>0.279693</td>
      <td>0.379200</td>
      <td>0.950869</td>
      <td>0.920629</td>
      <td>0.863200</td>
      <td>0.858960</td>
      <td>0.933914</td>
      <td>0.611007</td>
      <td>0.900950</td>
      <td>0.864385</td>
    </tr>
    <tr>
      <th>opencv</th>
      <td>0.823003</td>
      <td>0.825930</td>
      <td>0.669445</td>
      <td>0.826019</td>
      <td>0.858591</td>
      <td>0.802378</td>
      <td>0.900917</td>
      <td>0.783440</td>
      <td>0.665658</td>
      <td>0.677920</td>
      <td>0.819273</td>
      <td>0.546685</td>
      <td>0.658434</td>
      <td>0.805804</td>
      <td>1.000000</td>
      <td>0.733530</td>
      <td>0.858091</td>
      <td>0.869210</td>
      <td>0.721219</td>
      <td>0.802693</td>
      <td>-0.368552</td>
      <td>0.673537</td>
      <td>0.781996</td>
      <td>0.781829</td>
      <td>0.884038</td>
      <td>0.831157</td>
      <td>0.892362</td>
      <td>0.770546</td>
      <td>0.724499</td>
      <td>0.849124</td>
      <td>0.807656</td>
    </tr>
    <tr>
      <th>BeautifulSoup</th>
      <td>0.964120</td>
      <td>0.823409</td>
      <td>0.634024</td>
      <td>0.971518</td>
      <td>0.925860</td>
      <td>0.964891</td>
      <td>0.868467</td>
      <td>0.243028</td>
      <td>0.948894</td>
      <td>0.808054</td>
      <td>0.385708</td>
      <td>0.945113</td>
      <td>0.896149</td>
      <td>0.943511</td>
      <td>0.733530</td>
      <td>1.000000</td>
      <td>0.903434</td>
      <td>0.867146</td>
      <td>0.156589</td>
      <td>0.910534</td>
      <td>-0.645888</td>
      <td>0.139977</td>
      <td>0.241223</td>
      <td>0.912474</td>
      <td>0.868454</td>
      <td>0.829026</td>
      <td>0.790869</td>
      <td>0.913083</td>
      <td>0.536421</td>
      <td>0.870047</td>
      <td>0.841867</td>
    </tr>
    <tr>
      <th>scrapy</th>
      <td>0.938008</td>
      <td>0.899411</td>
      <td>0.769414</td>
      <td>0.943253</td>
      <td>0.963228</td>
      <td>0.941293</td>
      <td>0.943975</td>
      <td>0.530458</td>
      <td>0.853931</td>
      <td>0.771901</td>
      <td>0.667057</td>
      <td>0.764296</td>
      <td>0.844406</td>
      <td>0.902404</td>
      <td>0.858091</td>
      <td>0.903434</td>
      <td>1.000000</td>
      <td>0.886790</td>
      <td>0.455278</td>
      <td>0.884059</td>
      <td>-0.619175</td>
      <td>0.388894</td>
      <td>0.528629</td>
      <td>0.884349</td>
      <td>0.904389</td>
      <td>0.848008</td>
      <td>0.859776</td>
      <td>0.901858</td>
      <td>0.758921</td>
      <td>0.857091</td>
      <td>0.828177</td>
    </tr>
    <tr>
      <th>Pygame</th>
      <td>0.897563</td>
      <td>0.819469</td>
      <td>0.620004</td>
      <td>0.908709</td>
      <td>0.886094</td>
      <td>0.891209</td>
      <td>0.890109</td>
      <td>0.532474</td>
      <td>0.818508</td>
      <td>0.784766</td>
      <td>0.600705</td>
      <td>0.754918</td>
      <td>0.775641</td>
      <td>0.896524</td>
      <td>0.869210</td>
      <td>0.867146</td>
      <td>0.886790</td>
      <td>1.000000</td>
      <td>0.457291</td>
      <td>0.867885</td>
      <td>-0.524176</td>
      <td>0.440892</td>
      <td>0.531268</td>
      <td>0.859386</td>
      <td>0.894027</td>
      <td>0.840031</td>
      <td>0.856846</td>
      <td>0.859598</td>
      <td>0.611504</td>
      <td>0.876401</td>
      <td>0.800370</td>
    </tr>
    <tr>
      <th>Apache</th>
      <td>0.337570</td>
      <td>0.493519</td>
      <td>0.459789</td>
      <td>0.316988</td>
      <td>0.424820</td>
      <td>0.300755</td>
      <td>0.537619</td>
      <td>0.951598</td>
      <td>0.096910</td>
      <td>0.212115</td>
      <td>0.911161</td>
      <td>-0.105769</td>
      <td>0.157582</td>
      <td>0.286173</td>
      <td>0.721219</td>
      <td>0.156589</td>
      <td>0.455278</td>
      <td>0.457291</td>
      <td>1.000000</td>
      <td>0.346790</td>
      <td>0.022481</td>
      <td>0.869725</td>
      <td>0.951705</td>
      <td>0.286024</td>
      <td>0.475274</td>
      <td>0.465481</td>
      <td>0.553597</td>
      <td>0.304952</td>
      <td>0.627473</td>
      <td>0.431856</td>
      <td>0.372582</td>
    </tr>
    <tr>
      <th>Gunicorn</th>
      <td>0.934885</td>
      <td>0.851715</td>
      <td>0.682733</td>
      <td>0.941787</td>
      <td>0.919196</td>
      <td>0.927438</td>
      <td>0.887493</td>
      <td>0.416753</td>
      <td>0.879681</td>
      <td>0.751478</td>
      <td>0.530608</td>
      <td>0.832875</td>
      <td>0.833882</td>
      <td>0.926887</td>
      <td>0.802693</td>
      <td>0.910534</td>
      <td>0.884059</td>
      <td>0.867885</td>
      <td>0.346790</td>
      <td>1.000000</td>
      <td>-0.587320</td>
      <td>0.314476</td>
      <td>0.415339</td>
      <td>0.899108</td>
      <td>0.883061</td>
      <td>0.868798</td>
      <td>0.802585</td>
      <td>0.892897</td>
      <td>0.606029</td>
      <td>0.869428</td>
      <td>0.827829</td>
    </tr>
    <tr>
      <th>Pygtk</th>
      <td>-0.590748</td>
      <td>-0.571376</td>
      <td>-0.503671</td>
      <td>-0.617162</td>
      <td>-0.634004</td>
      <td>-0.648004</td>
      <td>-0.554801</td>
      <td>-0.088636</td>
      <td>-0.638582</td>
      <td>-0.463135</td>
      <td>-0.273881</td>
      <td>-0.620027</td>
      <td>-0.627745</td>
      <td>-0.611334</td>
      <td>-0.368552</td>
      <td>-0.645888</td>
      <td>-0.619175</td>
      <td>-0.524176</td>
      <td>0.022481</td>
      <td>-0.587320</td>
      <td>1.000000</td>
      <td>0.024193</td>
      <td>-0.087402</td>
      <td>-0.611068</td>
      <td>-0.556336</td>
      <td>-0.507529</td>
      <td>-0.445867</td>
      <td>-0.645867</td>
      <td>-0.442643</td>
      <td>-0.448065</td>
      <td>-0.419348</td>
    </tr>
    <tr>
      <th>Weka</th>
      <td>0.292909</td>
      <td>0.430621</td>
      <td>0.417443</td>
      <td>0.277733</td>
      <td>0.367647</td>
      <td>0.251633</td>
      <td>0.459200</td>
      <td>0.908043</td>
      <td>0.073342</td>
      <td>0.198937</td>
      <td>0.813893</td>
      <td>-0.091383</td>
      <td>0.104490</td>
      <td>0.279693</td>
      <td>0.673537</td>
      <td>0.139977</td>
      <td>0.388894</td>
      <td>0.440892</td>
      <td>0.869725</td>
      <td>0.314476</td>
      <td>0.024193</td>
      <td>1.000000</td>
      <td>0.908489</td>
      <td>0.281215</td>
      <td>0.475310</td>
      <td>0.439851</td>
      <td>0.536513</td>
      <td>0.279097</td>
      <td>0.551850</td>
      <td>0.406394</td>
      <td>0.345908</td>
    </tr>
    <tr>
      <th>matlab.1</th>
      <td>0.405177</td>
      <td>0.562623</td>
      <td>0.533660</td>
      <td>0.398713</td>
      <td>0.510440</td>
      <td>0.385120</td>
      <td>0.610998</td>
      <td>0.999949</td>
      <td>0.173679</td>
      <td>0.299685</td>
      <td>0.938936</td>
      <td>-0.014205</td>
      <td>0.220783</td>
      <td>0.379200</td>
      <td>0.781996</td>
      <td>0.241223</td>
      <td>0.528629</td>
      <td>0.531268</td>
      <td>0.951705</td>
      <td>0.415339</td>
      <td>-0.087402</td>
      <td>0.908489</td>
      <td>1.000000</td>
      <td>0.385194</td>
      <td>0.567146</td>
      <td>0.544125</td>
      <td>0.637689</td>
      <td>0.403539</td>
      <td>0.693942</td>
      <td>0.490245</td>
      <td>0.432119</td>
    </tr>
    <tr>
      <th>regression</th>
      <td>0.932747</td>
      <td>0.898775</td>
      <td>0.692117</td>
      <td>0.949591</td>
      <td>0.935454</td>
      <td>0.948927</td>
      <td>0.897000</td>
      <td>0.385995</td>
      <td>0.927352</td>
      <td>0.780885</td>
      <td>0.518338</td>
      <td>0.845012</td>
      <td>0.888283</td>
      <td>0.950869</td>
      <td>0.781829</td>
      <td>0.912474</td>
      <td>0.884349</td>
      <td>0.859386</td>
      <td>0.286024</td>
      <td>0.899108</td>
      <td>-0.611068</td>
      <td>0.281215</td>
      <td>0.385194</td>
      <td>1.000000</td>
      <td>0.916701</td>
      <td>0.847891</td>
      <td>0.844222</td>
      <td>0.919818</td>
      <td>0.627029</td>
      <td>0.871651</td>
      <td>0.835969</td>
    </tr>
    <tr>
      <th>classification</th>
      <td>0.912847</td>
      <td>0.920156</td>
      <td>0.735404</td>
      <td>0.926796</td>
      <td>0.938490</td>
      <td>0.920989</td>
      <td>0.922862</td>
      <td>0.568297</td>
      <td>0.869519</td>
      <td>0.771930</td>
      <td>0.667566</td>
      <td>0.737107</td>
      <td>0.863275</td>
      <td>0.920629</td>
      <td>0.884038</td>
      <td>0.868454</td>
      <td>0.904389</td>
      <td>0.894027</td>
      <td>0.475274</td>
      <td>0.883061</td>
      <td>-0.556336</td>
      <td>0.475310</td>
      <td>0.567146</td>
      <td>0.916701</td>
      <td>1.000000</td>
      <td>0.857105</td>
      <td>0.908351</td>
      <td>0.911334</td>
      <td>0.734111</td>
      <td>0.868255</td>
      <td>0.865458</td>
    </tr>
    <tr>
      <th>correlation</th>
      <td>0.883380</td>
      <td>0.850173</td>
      <td>0.681540</td>
      <td>0.887138</td>
      <td>0.885151</td>
      <td>0.875604</td>
      <td>0.883277</td>
      <td>0.544960</td>
      <td>0.786207</td>
      <td>0.718086</td>
      <td>0.622974</td>
      <td>0.719696</td>
      <td>0.751245</td>
      <td>0.863200</td>
      <td>0.831157</td>
      <td>0.829026</td>
      <td>0.848008</td>
      <td>0.840031</td>
      <td>0.465481</td>
      <td>0.868798</td>
      <td>-0.507529</td>
      <td>0.439851</td>
      <td>0.544125</td>
      <td>0.847891</td>
      <td>0.857105</td>
      <td>1.000000</td>
      <td>0.836347</td>
      <td>0.826589</td>
      <td>0.636888</td>
      <td>0.846042</td>
      <td>0.804044</td>
    </tr>
    <tr>
      <th>cluster-analysis</th>
      <td>0.856002</td>
      <td>0.843128</td>
      <td>0.703878</td>
      <td>0.863666</td>
      <td>0.888360</td>
      <td>0.850162</td>
      <td>0.883615</td>
      <td>0.638835</td>
      <td>0.758948</td>
      <td>0.715198</td>
      <td>0.697325</td>
      <td>0.653759</td>
      <td>0.733793</td>
      <td>0.858960</td>
      <td>0.892362</td>
      <td>0.790869</td>
      <td>0.859776</td>
      <td>0.856846</td>
      <td>0.553597</td>
      <td>0.802585</td>
      <td>-0.445867</td>
      <td>0.536513</td>
      <td>0.637689</td>
      <td>0.844222</td>
      <td>0.908351</td>
      <td>0.836347</td>
      <td>1.000000</td>
      <td>0.829776</td>
      <td>0.711633</td>
      <td>0.852380</td>
      <td>0.822327</td>
    </tr>
    <tr>
      <th>Sympy</th>
      <td>0.924760</td>
      <td>0.885348</td>
      <td>0.748211</td>
      <td>0.942827</td>
      <td>0.941951</td>
      <td>0.947783</td>
      <td>0.888094</td>
      <td>0.404499</td>
      <td>0.909197</td>
      <td>0.763999</td>
      <td>0.546557</td>
      <td>0.826870</td>
      <td>0.893532</td>
      <td>0.933914</td>
      <td>0.770546</td>
      <td>0.913083</td>
      <td>0.901858</td>
      <td>0.859598</td>
      <td>0.304952</td>
      <td>0.892897</td>
      <td>-0.645867</td>
      <td>0.279097</td>
      <td>0.403539</td>
      <td>0.919818</td>
      <td>0.911334</td>
      <td>0.826589</td>
      <td>0.829776</td>
      <td>1.000000</td>
      <td>0.694566</td>
      <td>0.837154</td>
      <td>0.809711</td>
    </tr>
    <tr>
      <th>stanford-nlp.1</th>
      <td>0.646442</td>
      <td>0.806042</td>
      <td>0.880015</td>
      <td>0.646793</td>
      <td>0.761613</td>
      <td>0.665346</td>
      <td>0.760325</td>
      <td>0.694404</td>
      <td>0.540140</td>
      <td>0.503749</td>
      <td>0.834700</td>
      <td>0.336663</td>
      <td>0.624416</td>
      <td>0.611007</td>
      <td>0.724499</td>
      <td>0.536421</td>
      <td>0.758921</td>
      <td>0.611504</td>
      <td>0.627473</td>
      <td>0.606029</td>
      <td>-0.442643</td>
      <td>0.551850</td>
      <td>0.693942</td>
      <td>0.627029</td>
      <td>0.734111</td>
      <td>0.636888</td>
      <td>0.711633</td>
      <td>0.694566</td>
      <td>1.000000</td>
      <td>0.566113</td>
      <td>0.602116</td>
    </tr>
    <tr>
      <th>pyqt</th>
      <td>0.910128</td>
      <td>0.815972</td>
      <td>0.644288</td>
      <td>0.896023</td>
      <td>0.871012</td>
      <td>0.870863</td>
      <td>0.868415</td>
      <td>0.491690</td>
      <td>0.797777</td>
      <td>0.703741</td>
      <td>0.569706</td>
      <td>0.769922</td>
      <td>0.738384</td>
      <td>0.900950</td>
      <td>0.849124</td>
      <td>0.870047</td>
      <td>0.857091</td>
      <td>0.876401</td>
      <td>0.431856</td>
      <td>0.869428</td>
      <td>-0.448065</td>
      <td>0.406394</td>
      <td>0.490245</td>
      <td>0.871651</td>
      <td>0.868255</td>
      <td>0.846042</td>
      <td>0.852380</td>
      <td>0.837154</td>
      <td>0.566113</td>
      <td>1.000000</td>
      <td>0.800720</td>
    </tr>
    <tr>
      <th>OCR</th>
      <td>0.866315</td>
      <td>0.798532</td>
      <td>0.624739</td>
      <td>0.876846</td>
      <td>0.853723</td>
      <td>0.859104</td>
      <td>0.829875</td>
      <td>0.432927</td>
      <td>0.832613</td>
      <td>0.776767</td>
      <td>0.514376</td>
      <td>0.749499</td>
      <td>0.806459</td>
      <td>0.864385</td>
      <td>0.807656</td>
      <td>0.841867</td>
      <td>0.828177</td>
      <td>0.800370</td>
      <td>0.372582</td>
      <td>0.827829</td>
      <td>-0.419348</td>
      <td>0.345908</td>
      <td>0.432119</td>
      <td>0.835969</td>
      <td>0.865458</td>
      <td>0.804044</td>
      <td>0.822327</td>
      <td>0.809711</td>
      <td>0.602116</td>
      <td>0.800720</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop(columns = ["matlab.1"])
```

### Analysing Variance

Thhe coefficient of variation for each column is calculated, in order to set a variance threshold. Since the feature of lowest variation has a coefficient of variation over 45, no features are dropped since variation is not too low.


```python
import numpy as np
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
pd.set_option('display.max_rows', 140)
listedOut = df.apply(cv)
```


```python
listedOut.sort_values()
```




    Apache               45.727337
    pyqt                 50.900401
    OCR                  51.070454
    cluster-analysis     54.261511
    opencv               54.453584
    month                58.395699
    matlab               60.758122
    scipy                61.963501
    classification       62.420371
    python               63.115124
    correlation          63.483488
    nltk                 67.965909
    Pygame               67.998622
    hadoop               68.688173
    nlp                  70.337566
    Pygtk                70.852412
    r                    71.097080
    Weka                 71.731508
    scrapy               74.032620
    time-series          74.913730
    numpy                76.405971
    Gunicorn             76.801798
    regression           78.896601
    BeautifulSoup        82.055862
    stanford-nlp.1       86.105313
    Sympy                86.627053
    stanford-nlp         89.361551
    neural-network       93.263944
    machine-learning     94.422853
    python-3.x          123.121068
    dtype: float64



### LASSO Regression
Since the dimensionality is still quite high, LASSO regression is used to calculate which variables have the least impact on the number of python questions asked. Grid search cross validation is used to find the best hyperparameters parameters.


```python
from sklearn.linear_model import Lasso
```


```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
```


```python
X = df.drop(columns = ["python"])
y = df["python"].to_frame()
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
```


```python
pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ("model",Lasso())
])
```


```python
search = GridSearchCV(pipeline,
                     {"model__alpha":np.arange(0.1,3,0.1)},
                     cv = 5,
                     scoring = "neg_mean_squared_error",
                     verbose=2
                     )
```


```python
search.fit(X_train, y_train)
```

    Fitting 5 folds for each of 29 candidates, totalling 145 fits
    [CV] END ...................................model__alpha=0.1; total time=   0.0s
    [CV] END ...................................model__alpha=0.1; total time=   0.0s
    [CV] END ...................................model__alpha=0.1; total time=   0.0s
    [CV] END ...................................model__alpha=0.1; total time=   0.0s
    [CV] END ...................................model__alpha=0.1; total time=   0.0s
    [CV] END ...................................model__alpha=0.2; total time=   0.0s
    [CV] END ...................................model__alpha=0.2; total time=   0.0s
    [CV] END ...................................model__alpha=0.2; total time=   0.0s
    [CV] END ...................................model__alpha=0.2; total time=   0.0s
    [CV] END ...................................model__alpha=0.2; total time=   0.0s
    [CV] END ...................model__alpha=0.30000000000000004; total time=   0.0s
    [CV] END ...................model__alpha=0.30000000000000004; total time=   0.0s
    [CV] END ...................model__alpha=0.30000000000000004; total time=   0.0s
    [CV] END ...................model__alpha=0.30000000000000004; total time=   0.0s
    [CV] END ...................model__alpha=0.30000000000000004; total time=   0.0s
    [CV] END ...................................model__alpha=0.4; total time=   0.0s
    [CV] END ...................................model__alpha=0.4; total time=   0.0s
    [CV] END ...................................model__alpha=0.4; total time=   0.0s
    [CV] END ...................................model__alpha=0.4; total time=   0.0s


    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.406e+06, tolerance: 3.296e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.447e+06, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.445e+06, tolerance: 3.095e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 9.297e+05, tolerance: 2.982e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.042e+06, tolerance: 3.202e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.112e+06, tolerance: 3.296e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.067e+06, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.013e+06, tolerance: 3.095e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 6.376e+05, tolerance: 2.982e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 7.386e+05, tolerance: 3.202e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 9.295e+05, tolerance: 3.296e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 8.283e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 7.548e+05, tolerance: 3.095e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.916e+05, tolerance: 2.982e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 5.529e+05, tolerance: 3.202e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 8.063e+05, tolerance: 3.296e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 6.679e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 5.866e+05, tolerance: 3.095e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.757e+05, tolerance: 2.982e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.334e+05, tolerance: 3.202e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 7.188e+05, tolerance: 3.296e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 5.665e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.712e+05, tolerance: 3.095e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.020e+05, tolerance: 2.982e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.549e+05, tolerance: 3.202e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 6.543e+05, tolerance: 3.296e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 5.231e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.917e+05, tolerance: 3.095e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 5.850e+05, tolerance: 3.296e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.934e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(


    [CV] END ...................................model__alpha=0.4; total time=   0.0s
    [CV] END ...................................model__alpha=0.5; total time=   0.0s
    [CV] END ...................................model__alpha=0.5; total time=   0.0s
    [CV] END ...................................model__alpha=0.5; total time=   0.0s
    [CV] END ...................................model__alpha=0.5; total time=   0.0s
    [CV] END ...................................model__alpha=0.5; total time=   0.0s
    [CV] END ...................................model__alpha=0.6; total time=   0.0s
    [CV] END ...................................model__alpha=0.6; total time=   0.0s
    [CV] END ...................................model__alpha=0.6; total time=   0.0s
    [CV] END ...................................model__alpha=0.6; total time=   0.0s
    [CV] END ...................................model__alpha=0.6; total time=   0.0s
    [CV] END ....................model__alpha=0.7000000000000001; total time=   0.0s
    [CV] END ....................model__alpha=0.7000000000000001; total time=   0.0s
    [CV] END ....................model__alpha=0.7000000000000001; total time=   0.0s
    [CV] END ....................model__alpha=0.7000000000000001; total time=   0.0s
    [CV] END ....................model__alpha=0.7000000000000001; total time=   0.0s
    [CV] END ...................................model__alpha=0.8; total time=   0.0s
    [CV] END ...................................model__alpha=0.8; total time=   0.0s
    [CV] END ...................................model__alpha=0.8; total time=   0.0s
    [CV] END ...................................model__alpha=0.8; total time=   0.0s
    [CV] END ...................................model__alpha=0.8; total time=   0.0s
    [CV] END ...................................model__alpha=0.9; total time=   0.0s
    [CV] END ...................................model__alpha=0.9; total time=   0.0s
    [CV] END ...................................model__alpha=0.9; total time=   0.0s
    [CV] END ...................................model__alpha=0.9; total time=   0.0s
    [CV] END ...................................model__alpha=0.9; total time=   0.0s
    [CV] END ...................................model__alpha=1.0; total time=   0.0s
    [CV] END ...................................model__alpha=1.0; total time=   0.0s
    [CV] END ...................................model__alpha=1.0; total time=   0.0s


    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.958e+05, tolerance: 3.296e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.696e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.110e+05, tolerance: 3.296e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.492e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.350e+05, tolerance: 3.296e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.317e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(


    [CV] END ...................................model__alpha=1.0; total time=   0.0s
    [CV] END ...................................model__alpha=1.0; total time=   0.0s
    [CV] END ...................................model__alpha=1.1; total time=   0.0s
    [CV] END ...................................model__alpha=1.1; total time=   0.0s
    [CV] END ...................................model__alpha=1.1; total time=   0.0s
    [CV] END ...................................model__alpha=1.1; total time=   0.0s
    [CV] END ...................................model__alpha=1.1; total time=   0.0s
    [CV] END ....................model__alpha=1.2000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.2000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.2000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.2000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.2000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.3000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=1.3000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=1.3000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=1.3000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=1.3000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=1.4000000000000001; total time=   0.0s


    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.162e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.019e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.888e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.769e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(


    [CV] END ....................model__alpha=1.4000000000000001; total time=   0.0s
    [CV] END ....................model__alpha=1.4000000000000001; total time=   0.0s
    [CV] END ....................model__alpha=1.4000000000000001; total time=   0.0s
    [CV] END ....................model__alpha=1.4000000000000001; total time=   0.0s
    [CV] END ....................model__alpha=1.5000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.5000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.5000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.5000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.5000000000000002; total time=   0.0s
    [CV] END ...................................model__alpha=1.6; total time=   0.0s
    [CV] END ...................................model__alpha=1.6; total time=   0.0s
    [CV] END ...................................model__alpha=1.6; total time=   0.0s
    [CV] END ...................................model__alpha=1.6; total time=   0.0s


    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.657e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.555e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(
    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.456e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(


    [CV] END ...................................model__alpha=1.6; total time=   0.0s
    [CV] END ....................model__alpha=1.7000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.7000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.7000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.7000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.7000000000000002; total time=   0.0s
    [CV] END ....................model__alpha=1.8000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=1.8000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=1.8000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=1.8000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=1.8000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=1.9000000000000001; total time=   0.0s
    [CV] END ....................model__alpha=1.9000000000000001; total time=   0.0s
    [CV] END ....................model__alpha=1.9000000000000001; total time=   0.0s
    [CV] END ....................model__alpha=1.9000000000000001; total time=   0.0s
    [CV] END ....................model__alpha=1.9000000000000001; total time=   0.0s
    [CV] END ...................................model__alpha=2.0; total time=   0.0s
    [CV] END ...................................model__alpha=2.0; total time=   0.0s
    [CV] END ...................................model__alpha=2.0; total time=   0.0s
    [CV] END ...................................model__alpha=2.0; total time=   0.0s


    /home/ec2-user/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.351e+05, tolerance: 3.163e+05
      model = cd_fast.enet_coordinate_descent(


    [CV] END ...................................model__alpha=2.0; total time=   0.0s
    [CV] END ...................................model__alpha=2.1; total time=   0.0s
    [CV] END ...................................model__alpha=2.1; total time=   0.0s
    [CV] END ...................................model__alpha=2.1; total time=   0.0s
    [CV] END ...................................model__alpha=2.1; total time=   0.0s
    [CV] END ...................................model__alpha=2.1; total time=   0.0s
    [CV] END ...................................model__alpha=2.2; total time=   0.0s
    [CV] END ...................................model__alpha=2.2; total time=   0.0s
    [CV] END ...................................model__alpha=2.2; total time=   0.0s
    [CV] END ...................................model__alpha=2.2; total time=   0.0s
    [CV] END ...................................model__alpha=2.2; total time=   0.0s
    [CV] END ....................model__alpha=2.3000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=2.3000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=2.3000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=2.3000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=2.3000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=2.4000000000000004; total time=   0.0s
    [CV] END ....................model__alpha=2.4000000000000004; total time=   0.0s
    [CV] END ....................model__alpha=2.4000000000000004; total time=   0.0s
    [CV] END ....................model__alpha=2.4000000000000004; total time=   0.0s
    [CV] END ....................model__alpha=2.4000000000000004; total time=   0.0s
    [CV] END ....................model__alpha=2.5000000000000004; total time=   0.0s
    [CV] END ....................model__alpha=2.5000000000000004; total time=   0.0s
    [CV] END ....................model__alpha=2.5000000000000004; total time=   0.0s
    [CV] END ....................model__alpha=2.5000000000000004; total time=   0.0s
    [CV] END ....................model__alpha=2.5000000000000004; total time=   0.0s
    [CV] END ...................................model__alpha=2.6; total time=   0.0s
    [CV] END ...................................model__alpha=2.6; total time=   0.0s
    [CV] END ...................................model__alpha=2.6; total time=   0.0s
    [CV] END ...................................model__alpha=2.6; total time=   0.0s
    [CV] END ...................................model__alpha=2.6; total time=   0.0s
    [CV] END ...................................model__alpha=2.7; total time=   0.0s
    [CV] END ...................................model__alpha=2.7; total time=   0.0s
    [CV] END ...................................model__alpha=2.7; total time=   0.0s
    [CV] END ...................................model__alpha=2.7; total time=   0.0s
    [CV] END ...................................model__alpha=2.7; total time=   0.0s
    [CV] END ....................model__alpha=2.8000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=2.8000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=2.8000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=2.8000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=2.8000000000000003; total time=   0.0s
    [CV] END ....................model__alpha=2.9000000000000004; total time=   0.0s
    [CV] END ....................model__alpha=2.9000000000000004; total time=   0.0s
    [CV] END ....................model__alpha=2.9000000000000004; total time=   0.0s
    [CV] END ....................model__alpha=2.9000000000000004; total time=   0.0s
    [CV] END ....................model__alpha=2.9000000000000004; total time=   0.0s





    GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[('scaler', StandardScaler()),
                                           ('model', Lasso())]),
                 param_grid={'model__alpha': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3,
           1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6,
           2.7, 2.8, 2.9])},
                 scoring='neg_mean_squared_error', verbose=2)




```python
search.best_params_
```




    {'model__alpha': 2.9000000000000004}




```python
coef = search.best_estimator_[1].coef_
```


```python
np.array(X.columns)[coef == 0]
```




    array(['nltk', 'machine-learning'], dtype=object)




```python
df = df.drop(columns = ['nltk', 'machine-learning'])
```

The number of features has now been reduced from 82 to 28. A significant dimensionality reduction.


```python
len(df.columns)
```




    28




```python
df.to_csv("features_selected.csv", index = False)
```

```python
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
```


```python
df = pd.read_csv("features_selected.csv")
```

I removed the month column as it will serve no purpose, but increase dimensionality.


```python
df = df.drop(columns = ["month"])
df.head()
```




<div style="overflow-x:auto;">
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
      <th>stanford-nlp</th>
      <th>python</th>
      <th>r</th>
      <th>numpy</th>
      <th>scipy</th>
      <th>matlab</th>
      <th>nlp</th>
      <th>hadoop</th>
      <th>python-3.x</th>
      <th>neural-network</th>
      <th>time-series</th>
      <th>opencv</th>
      <th>BeautifulSoup</th>
      <th>scrapy</th>
      <th>Pygame</th>
      <th>Apache</th>
      <th>Gunicorn</th>
      <th>Pygtk</th>
      <th>Weka</th>
      <th>regression</th>
      <th>classification</th>
      <th>correlation</th>
      <th>cluster-analysis</th>
      <th>Sympy</th>
      <th>stanford-nlp.1</th>
      <th>pyqt</th>
      <th>OCR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>631</td>
      <td>8</td>
      <td>6</td>
      <td>2</td>
      <td>19</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>95</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>633</td>
      <td>9</td>
      <td>7</td>
      <td>3</td>
      <td>27</td>
      <td>9</td>
      <td>1</td>
      <td>12</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>114</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>766</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>24</td>
      <td>12</td>
      <td>1</td>
      <td>4</td>
      <td>11</td>
      <td>1</td>
      <td>7</td>
      <td>8</td>
      <td>0</td>
      <td>7</td>
      <td>104</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>768</td>
      <td>12</td>
      <td>6</td>
      <td>3</td>
      <td>32</td>
      <td>14</td>
      <td>6</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>111</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1003</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>42</td>
      <td>9</td>
      <td>3</td>
      <td>19</td>
      <td>7</td>
      <td>0</td>
      <td>10</td>
      <td>14</td>
      <td>0</td>
      <td>2</td>
      <td>127</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



# Data Preparation And Standardisation
Much of the code below was lifted from [here](https://www.tensorflow.org/tutorials/structured_data/time_series#1_indexes_and_offsets). 

I did a train/test/validation split here of 80-10-10 which has as large a training dataset as possible due to the still high dimensionality.


```python
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)
train_df = df[0:int(n*0.8)]
val_df = df[int(n*0.8):int(n*0.9)]
test_df = df[int(n*0.9):]
```

I standardised the three datasets.


```python
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
```


```python
pd.set_option('display.max_columns', None)
train_df.head()
```




<div style="overflow-x:auto;">
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
      <th>stanford-nlp</th>
      <th>python</th>
      <th>r</th>
      <th>numpy</th>
      <th>scipy</th>
      <th>matlab</th>
      <th>nlp</th>
      <th>hadoop</th>
      <th>python-3.x</th>
      <th>neural-network</th>
      <th>time-series</th>
      <th>opencv</th>
      <th>BeautifulSoup</th>
      <th>scrapy</th>
      <th>Pygame</th>
      <th>Apache</th>
      <th>Gunicorn</th>
      <th>Pygtk</th>
      <th>Weka</th>
      <th>regression</th>
      <th>classification</th>
      <th>correlation</th>
      <th>cluster-analysis</th>
      <th>Sympy</th>
      <th>stanford-nlp.1</th>
      <th>pyqt</th>
      <th>OCR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.029562</td>
      <td>-1.444048</td>
      <td>-1.235520</td>
      <td>-1.182943</td>
      <td>-1.402082</td>
      <td>-1.564972</td>
      <td>-1.473996</td>
      <td>-1.371492</td>
      <td>-0.846017</td>
      <td>-0.851402</td>
      <td>-1.268088</td>
      <td>-1.576239</td>
      <td>-1.155275</td>
      <td>-1.154736</td>
      <td>-1.352202</td>
      <td>-1.892737</td>
      <td>-1.216528</td>
      <td>-1.433769</td>
      <td>-1.403838</td>
      <td>-1.147215</td>
      <td>-1.367075</td>
      <td>-1.44988</td>
      <td>-1.618861</td>
      <td>-0.943702</td>
      <td>-1.029562</td>
      <td>-1.853989</td>
      <td>-1.567741</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.029562</td>
      <td>-1.443636</td>
      <td>-1.234864</td>
      <td>-1.179755</td>
      <td>-1.386856</td>
      <td>-1.546378</td>
      <td>-1.448732</td>
      <td>-1.371492</td>
      <td>-0.835600</td>
      <td>-0.851402</td>
      <td>-1.242006</td>
      <td>-1.584915</td>
      <td>-1.168722</td>
      <td>-1.154736</td>
      <td>-1.275435</td>
      <td>-1.830777</td>
      <td>-1.216528</td>
      <td>-1.067700</td>
      <td>-1.403838</td>
      <td>-1.073337</td>
      <td>-1.285609</td>
      <td>-1.44988</td>
      <td>-1.618861</td>
      <td>-1.007589</td>
      <td>-1.029562</td>
      <td>-1.853989</td>
      <td>-1.149234</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.029562</td>
      <td>-1.416248</td>
      <td>-1.238143</td>
      <td>-1.189319</td>
      <td>-1.402082</td>
      <td>-1.553351</td>
      <td>-1.372938</td>
      <td>-1.371492</td>
      <td>-0.846017</td>
      <td>-0.768042</td>
      <td>-1.268088</td>
      <td>-1.558886</td>
      <td>-1.114934</td>
      <td>-1.154736</td>
      <td>-1.217859</td>
      <td>-1.863388</td>
      <td>-1.216528</td>
      <td>-1.159217</td>
      <td>-1.403838</td>
      <td>-1.184153</td>
      <td>-1.448541</td>
      <td>-1.44988</td>
      <td>-1.449389</td>
      <td>-1.007589</td>
      <td>-1.029562</td>
      <td>-1.803055</td>
      <td>-1.776994</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.029562</td>
      <td>-1.415836</td>
      <td>-1.232896</td>
      <td>-1.182943</td>
      <td>-1.386856</td>
      <td>-1.534757</td>
      <td>-1.322410</td>
      <td>-1.350084</td>
      <td>-0.838204</td>
      <td>-0.827585</td>
      <td>-1.294171</td>
      <td>-1.554548</td>
      <td>-1.182169</td>
      <td>-1.154736</td>
      <td>-1.237051</td>
      <td>-1.840560</td>
      <td>-1.216528</td>
      <td>-1.433769</td>
      <td>-1.403838</td>
      <td>-1.184153</td>
      <td>-1.285609</td>
      <td>-1.44988</td>
      <td>-1.562371</td>
      <td>-1.007589</td>
      <td>-1.029562</td>
      <td>-1.701187</td>
      <td>-1.567741</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.029562</td>
      <td>-1.367444</td>
      <td>-1.239455</td>
      <td>-1.179755</td>
      <td>-1.402082</td>
      <td>-1.511516</td>
      <td>-1.448732</td>
      <td>-1.362929</td>
      <td>-0.826484</td>
      <td>-0.815676</td>
      <td>-1.294171</td>
      <td>-1.545872</td>
      <td>-1.034253</td>
      <td>-1.154736</td>
      <td>-1.313818</td>
      <td>-1.788383</td>
      <td>-1.216528</td>
      <td>-1.250734</td>
      <td>-1.403838</td>
      <td>-1.110276</td>
      <td>-1.448541</td>
      <td>-1.44988</td>
      <td>-1.336408</td>
      <td>-1.007589</td>
      <td>-1.029562</td>
      <td>-1.726654</td>
      <td>-1.707243</td>
    </tr>
  </tbody>
</table>
</div>



# Windowing
The class that generates a windowing scheme for the data is below. The data is then windowed with three preceeding predictors to one target variable. 


```python
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
```


```python
window = WindowGenerator(input_width=3, label_width=1, shift=1,
                     label_columns=["python"])
window
```




    Total window size: 4
    Input indices: [0 1 2]
    Label indices: [3]
    Label column name(s): ['python']



Here, the windowing scheme is applied to the data


```python
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window
```


```python
example_window = tf.stack([np.array(train_df[:window.total_window_size]),
                           np.array(train_df[20:20+window.total_window_size]),
                           np.array(train_df[60:60+window.total_window_size])])

example_inputs, example_labels = window.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

```

    All shapes are: (batch, time, features)
    Window shape: (3, 4, 27)
    Inputs shape: (3, 3, 27)
    Labels shape: (3, 1, 1)


Here, three example windows are visualised, with the three predictors and the target.


```python
window.example = example_inputs, example_labels
def plot(self, model=None, plot_col='python', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('months')

WindowGenerator.plot = plot

```


```python
window.plot()
```


![png](/images/time-series-data/output_17_0.png)


Here the windows are combined to make one dataset for the models.


```python
def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset
```


```python
@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
```


```python
window.train.element_spec
```




    (TensorSpec(shape=(None, 3, 27), dtype=tf.float32, name=None),
     TensorSpec(shape=(None, 1, 1), dtype=tf.float32, name=None))



# Model Training
## Baseline Model

Here the baseline model needs a single step window to be trained. It will simply repeat the last value to predict the next. I expect high performance as the data takes a broadly linear form with a very slight incline.


```python
single_step = WindowGenerator(input_width=1, label_width=1, shift=1,
                     label_columns=["python"])
for example_inputs, example_labels in single_step.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
```

    Inputs shape (batch, time, features): (32, 1, 27)
    Labels shape (batch, time, features): (32, 1, 1)



```python
class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index
    
    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
```


```python
baseline = Baseline(label_index = column_indices["python"])
baseline.compile(loss = tf.keras.losses.MeanSquaredError(), metrics = [tf.keras.metrics.MeanAbsoluteError()])
```


```python
val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step.val)
performance['Baseline'] = baseline.evaluate(single_step.test, verbose=0)
```

    1/1 [==============================] - 0s 108ms/step - loss: 0.0735 - mean_absolute_error: 0.2085



```python
single_step
```




    Total window size: 2
    Input indices: [0]
    Label indices: [1]
    Label column name(s): ['python']




```python
wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['python'])

wide_window
```




    Total window size: 25
    Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
    Label indices: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
    Label column name(s): ['python']




```python
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)
```

    Input shape: (32, 24, 27)
    Output shape: (32, 24, 1)



```python
wide_window.plot(baseline)
```


![png](/images/time-series-data/output_30_0.png)


That performed as well as imagined.

## Linear Model

If the data is not too high dimensional, then the linear model will significantly outperform the baseline.


```python
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])
print('Input shape:', single_step.example[0].shape)
print('Output shape:', linear(single_step.example[0]).shape)
```

    Input shape: (32, 1, 27)
    Output shape: (32, 1, 1)



```python
MAX_EPOCHS = 500

def compile_and_fit(model, window, patience):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history
```


```python
history = compile_and_fit(linear, single_step, patience=15)

val_performance['Linear'] = linear.evaluate(single_step.val)
performance['Linear'] = linear.evaluate(single_step.test, verbose=0)
```

    Epoch 1/500
    4/4 [==============================] - 1s 58ms/step - loss: 3.1680 - mean_absolute_error: 1.5870 - val_loss: 12.5611 - val_mean_absolute_error: 3.5082
    Epoch 2/500
    4/4 [==============================] - 0s 55ms/step - loss: 2.8518 - mean_absolute_error: 1.5029 - val_loss: 11.5955 - val_mean_absolute_error: 3.3677
    Epoch 3/500
    4/4 [==============================] - 0s 22ms/step - loss: 2.5615 - mean_absolute_error: 1.4206 - val_loss: 10.6727 - val_mean_absolute_error: 3.2276
    Epoch 4/500
    4/4 [==============================] - 0s 24ms/step - loss: 2.2760 - mean_absolute_error: 1.3390 - val_loss: 9.8141 - val_mean_absolute_error: 3.0914
    Epoch 5/500
    4/4 [==============================] - 0s 20ms/step - loss: 2.0348 - mean_absolute_error: 1.2632 - val_loss: 8.9995 - val_mean_absolute_error: 2.9563
    ...
    Epoch 330/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.0697 - mean_absolute_error: 0.1937 - val_loss: 0.2184 - val_mean_absolute_error: 0.4213
    Epoch 331/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.0695 - mean_absolute_error: 0.1933 - val_loss: 0.2179 - val_mean_absolute_error: 0.4207
    Epoch 332/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.0692 - mean_absolute_error: 0.1930 - val_loss: 0.2202 - val_mean_absolute_error: 0.4226
    Epoch 333/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.0690 - mean_absolute_error: 0.1933 - val_loss: 0.2209 - val_mean_absolute_error: 0.4231
    Epoch 334/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.0687 - mean_absolute_error: 0.1931 - val_loss: 0.2184 - val_mean_absolute_error: 0.4209
    1/1 [==============================] - 0s 49ms/step - loss: 0.2184 - mean_absolute_error: 0.4209



```python
wide_window.plot(linear)
```


![png](/images/time-series-data/output_35_0.png)


That significantly underperforms compared to what ought to be expected.


```python
plt.bar(x = range(len(train_df.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)
```


![png](/images/time-series-data/output_37_0.png)


As can be seen here, python has a medium weight, where it has a clear positive trend that would be expected to give it a higher weight than all other features. This is likely a case of the random state impacting the outcome. This should establish linear models as unsuitable for multiregression.

## Dense Neural Network, Single Step

This is a relu-activated neural network (accounts for nonlinearities). I expect this to outperform the linear model since the random state will not have so much control, and relations between data points will be taken into account by the hidden layers, hopefully giving python more sway.


```python
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(dense, single_step, patience = 10)

val_performance['Dense'] = dense.evaluate(single_step.val)
performance['Dense'] = dense.evaluate(single_step.test, verbose=0)
```

    Epoch 1/500
    4/4 [==============================] - 1s 54ms/step - loss: 0.9164 - mean_absolute_error: 0.8034 - val_loss: 0.3702 - val_mean_absolute_error: 0.4670
    Epoch 2/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.3167 - mean_absolute_error: 0.4421 - val_loss: 0.2078 - val_mean_absolute_error: 0.4003
    Epoch 3/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.1174 - mean_absolute_error: 0.2724 - val_loss: 0.2033 - val_mean_absolute_error: 0.3922
    Epoch 4/500
    4/4 [==============================] - 0s 21ms/step - loss: 0.0608 - mean_absolute_error: 0.1875 - val_loss: 0.2013 - val_mean_absolute_error: 0.3413
    Epoch 5/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.0773 - mean_absolute_error: 0.2353 - val_loss: 0.2382 - val_mean_absolute_error: 0.3576
    ...
    Epoch 50/500
    4/4 [==============================] - 0s 21ms/step - loss: 0.0038 - mean_absolute_error: 0.0493 - val_loss: 0.0771 - val_mean_absolute_error: 0.2417
    Epoch 51/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.0037 - mean_absolute_error: 0.0485 - val_loss: 0.0763 - val_mean_absolute_error: 0.2393
    Epoch 52/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.0036 - mean_absolute_error: 0.0473 - val_loss: 0.0791 - val_mean_absolute_error: 0.2440
    Epoch 53/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.0035 - mean_absolute_error: 0.0462 - val_loss: 0.0728 - val_mean_absolute_error: 0.2301
    Epoch 54/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.0034 - mean_absolute_error: 0.0456 - val_loss: 0.0725 - val_mean_absolute_error: 0.2299
    1/1 [==============================] - 0s 51ms/step - loss: 0.0725 - mean_absolute_error: 0.2299



```python
wide_window.plot(dense)
```


![png](/images/time-series-data/output_40_0.png)


It has a performance very similar to the baseline and appears to be fitting quite well to teh data. Perhaps if given more training data it would significantly outperform the baseline.

## Dense Neural Network, Multi-Step

This uses the three step window discussed earlier. The increased dimensionality from having more steps in the window may cause overfitting, but may also provide extra relevant data for prediction.


```python
multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

history = compile_and_fit(multi_step_dense, window, patience  = 15)

val_performance['Dense multistep'] = multi_step_dense.evaluate(window.val)
performance['Dense multistep'] = multi_step_dense.evaluate(window.test, verbose=0)
```

    Epoch 1/500
    4/4 [==============================] - 0s 42ms/step - loss: 1.9811 - mean_absolute_error: 1.1725 - val_loss: 0.1074 - val_mean_absolute_error: 0.2554
    Epoch 2/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.3816 - mean_absolute_error: 0.5119 - val_loss: 0.5636 - val_mean_absolute_error: 0.6652
    Epoch 3/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.1567 - mean_absolute_error: 0.2961 - val_loss: 0.4896 - val_mean_absolute_error: 0.6174
    Epoch 4/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.1458 - mean_absolute_error: 0.3221 - val_loss: 0.1449 - val_mean_absolute_error: 0.3090
    Epoch 5/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.1302 - mean_absolute_error: 0.3152 - val_loss: 0.0868 - val_mean_absolute_error: 0.2272
    ...
    Epoch 31/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.0146 - mean_absolute_error: 0.0934 - val_loss: 0.0858 - val_mean_absolute_error: 0.2360
    Epoch 32/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.0140 - mean_absolute_error: 0.0919 - val_loss: 0.0756 - val_mean_absolute_error: 0.2181
    Epoch 33/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.0134 - mean_absolute_error: 0.0900 - val_loss: 0.0714 - val_mean_absolute_error: 0.2120
    Epoch 34/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.0130 - mean_absolute_error: 0.0883 - val_loss: 0.0800 - val_mean_absolute_error: 0.2259
    Epoch 35/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.0125 - mean_absolute_error: 0.0866 - val_loss: 0.0739 - val_mean_absolute_error: 0.2164
    1/1 [==============================] - 0s 51ms/step - loss: 0.0739 - mean_absolute_error: 0.2164



```python
window.plot(multi_step_dense)
```


![png](/images/time-series-data/output_43_0.png)


The increased dimensionality from seeing more of the dataset causes clear overfitting with the difference in loss between training and validation MAE.
## Convolutional Neural Network

Since the multi-step dense model shows signs of overfitting, the convolutional neural netowork can be expected to do the same as it receives the same data.


```python
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(3,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

```


```python
print("Conv model on `window`")
print('Input shape:', window.example[0].shape)
print('Output shape:', conv_model(window.example[0]).shape)
```

    Conv model on `window`
    Input shape: (32, 3, 27)
    Output shape: (32, 1, 1)



```python
history = compile_and_fit(conv_model, window, patience = 15)


val_performance['Conv'] = conv_model.evaluate(window.val)
performance['Conv'] = conv_model.evaluate(window.test, verbose=0)
```

    Epoch 1/500
    4/4 [==============================] - 1s 51ms/step - loss: 1.9010 - mean_absolute_error: 1.2026 - val_loss: 2.6069 - val_mean_absolute_error: 1.5940
    Epoch 2/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.7829 - mean_absolute_error: 0.7768 - val_loss: 0.7883 - val_mean_absolute_error: 0.8420
    Epoch 3/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.2806 - mean_absolute_error: 0.4620 - val_loss: 0.1224 - val_mean_absolute_error: 0.2916
    Epoch 4/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.1104 - mean_absolute_error: 0.2657 - val_loss: 0.1048 - val_mean_absolute_error: 0.2514
    ...
    Epoch 51/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.0063 - mean_absolute_error: 0.0588 - val_loss: 0.0373 - val_mean_absolute_error: 0.1671
    Epoch 52/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.0067 - mean_absolute_error: 0.0607 - val_loss: 0.0370 - val_mean_absolute_error: 0.1627
    Epoch 53/500
    4/4 [==============================] - 0s 20ms/step - loss: 0.0058 - mean_absolute_error: 0.0571 - val_loss: 0.0378 - val_mean_absolute_error: 0.1540
    Epoch 54/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.0061 - mean_absolute_error: 0.0567 - val_loss: 0.0395 - val_mean_absolute_error: 0.1468
    Epoch 55/500
    4/4 [==============================] - 0s 19ms/step - loss: 0.0057 - mean_absolute_error: 0.0549 - val_loss: 0.0368 - val_mean_absolute_error: 0.1537
    1/1 [==============================] - 0s 49ms/step - loss: 0.0368 - mean_absolute_error: 0.1537



```python
LABEL_WIDTH = 24
CONV_WIDTH = 3
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['python'])

wide_conv_window
```




    Total window size: 27
    Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25]
    Label indices: [ 3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26]
    Label column name(s): ['python']




```python
wide_conv_window.plot(conv_model)
```


![png](/images/time-series-data/output_49_0.png)



```python
print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)
```

    Wide conv window
    Input shape: (32, 26, 27)
    Labels shape: (32, 24, 1)
    Output shape: (32, 24, 1)


# Final Assessment And Other Datasets

The convolutional model not only outperforms the baseline, but significantly outperforms it on the test data which is the furthest from the training dataset as far as time is concerned. This shows that it is a very useful model that will continue to perform well on whatever future data emerges.


```python
x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = linear.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [python, normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
```


![png](/images/time-series-data/output_52_0.png)



```python
for name, value in performance.items():
  print(f'{name:12s}: {value[1]:0.4f}')
```

    Baseline    : 0.3429
    Linear      : 1.1753
    Dense       : 0.2958
    Dense multistep: 0.4550
    Conv        : 0.4278


## Models Exclusively Trained on Python
As the almost linear relationship between python questions and the month is clear. I decided to test these models on a dataset of only python data, with a window size of 6 since the dimensionality will still be low as there is only one feature. It attained these results:

![png](/images/time-series-data/final graph.png)
This is the convolutional network's graph. Notice the smoothness as a result of its inflexibility. This limits overfitting. Since there are no other variables present, this model will be resilient to future changes in the other libraries usage. This is clearly the best model for prediction.
![png](/images/time-series-data/output_38_0.png)
