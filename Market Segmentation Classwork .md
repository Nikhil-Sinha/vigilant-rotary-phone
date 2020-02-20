## Import the Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set

from sklearn.cluster import KMeans
```

## Load the Data



```python
data = pd.read_csv('3.12. Example.csv')
data
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
      <th>Satisfaction</th>
      <th>Loyalty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>-1.33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>-0.28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>-0.99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>-0.29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>-1.66</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>-0.97</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>-0.32</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>1.02</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>-0.34</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>-1.69</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>14</th>
      <td>7</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>15</th>
      <td>9</td>
      <td>1.36</td>
    </tr>
    <tr>
      <th>16</th>
      <td>8</td>
      <td>1.38</td>
    </tr>
    <tr>
      <th>17</th>
      <td>7</td>
      <td>1.36</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7</td>
      <td>-0.34</td>
    </tr>
    <tr>
      <th>19</th>
      <td>9</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3</td>
      <td>-1.69</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3</td>
      <td>-0.96</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6</td>
      <td>1.03</td>
    </tr>
    <tr>
      <th>25</th>
      <td>9</td>
      <td>-0.99</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>27</th>
      <td>9</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3</td>
      <td>-1.36</td>
    </tr>
    <tr>
      <th>29</th>
      <td>5</td>
      <td>0.73</td>
    </tr>
  </tbody>
</table>
</div>



#### here "Satisfaction" is Discrete and an Integer. Self given value.

#### "Loyality" is a Continuous and ranges from (-2.5 to 2.5)

## Plot the data 


```python
plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyality')
plt.show()
```


![png](output_7_0.png)


### We have to have 4 clusters as 2 clusters will not give an appropirate result

## Cluster


```python
x = data.copy()
kmeans = KMeans(2)
kmeans.fit(x)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=2, n_init=10, n_jobs=None, precompute_distances='auto',
           random_state=None, tol=0.0001, verbose=0)



## Cluster Result


```python
clusters = x.copy()
clusters['pred'] = kmeans.fit_predict(x)
clusters
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
      <th>Satisfaction</th>
      <th>Loyalty</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>-1.33</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>-0.28</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>-0.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>-0.29</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.06</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>-1.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>-0.97</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>-0.32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>1.02</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8</td>
      <td>0.68</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>-0.34</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5</td>
      <td>0.39</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>-1.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2</td>
      <td>0.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>7</td>
      <td>0.27</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>9</td>
      <td>1.36</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>8</td>
      <td>1.38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>7</td>
      <td>1.36</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7</td>
      <td>-0.34</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>9</td>
      <td>0.67</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10</td>
      <td>1.18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3</td>
      <td>-1.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4</td>
      <td>1.04</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3</td>
      <td>-0.96</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6</td>
      <td>1.03</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>9</td>
      <td>-0.99</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10</td>
      <td>0.37</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>9</td>
      <td>0.03</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3</td>
      <td>-1.36</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>5</td>
      <td>0.73</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.scatter(clusters['Satisfaction'],clusters['Loyalty'],c = clusters['pred'],cmap = 'rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyality')
plt.show()
```


![png](output_13_0.png)


### In the above graph, we can see that the graph is split in half. 

### Only "Satisfaction" is considered and "Loyalty" is disregarded. We need to Standardize now.

## Standardize the Variables


```python
from sklearn import preprocessing
```


```python
x_scaled = preprocessing.scale(x)
x_scaled
```




    array([[-0.93138063, -1.3318111 ],
           [-0.15523011, -0.28117124],
           [-0.54330537, -0.99160391],
           [ 0.23284516, -0.29117733],
           [-0.93138063,  1.05964534],
           [-2.09560642, -1.6620122 ],
           [ 1.39707095, -0.97159172],
           [ 0.62092042, -0.32119561],
           [ 0.62092042,  1.01962097],
           [ 0.62092042,  0.67941378],
           [ 1.39707095, -0.3412078 ],
           [-0.54330537,  0.38923705],
           [-0.54330537, -1.69203048],
           [-1.70753116,  0.66940768],
           [ 0.23284516,  0.26916393],
           [ 1.00899568,  1.35982816],
           [ 0.62092042,  1.37984035],
           [ 0.23284516,  1.35982816],
           [ 0.23284516, -0.3412078 ],
           [ 1.00899568,  0.66940768],
           [ 1.39707095,  1.17971847],
           [-1.31945589, -1.69203048],
           [-0.93138063,  1.03963316],
           [-1.31945589, -0.96158562],
           [-0.15523011,  1.02962706],
           [ 1.00899568, -0.99160391],
           [ 1.39707095,  0.36922486],
           [ 1.00899568,  0.02901767],
           [-1.31945589, -1.36182938],
           [-0.54330537,  0.72944425]])



#### The .scale(x) scales each variable/column separately

#### As we dont know the number of clusters needed. We use Elbow Method


```python
wcss=[]
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

```


```python
wcss
```




    [60.0,
     29.818973034723147,
     17.913349527387965,
     10.247181805928422,
     7.792695153937187,
     6.569489487091783,
     5.538868131545409,
     4.380320178840311,
     3.7523551963246464]




```python
plt.plot(range(1,10),wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
```




    Text(0, 0.5, 'WCSS')




![png](output_23_1.png)


## Explore Clustering Solutions and select the number of Clusters


```python
kmeans_new = KMeans(4)
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)
```


```python
clusters_new
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
      <th>Satisfaction</th>
      <th>Loyalty</th>
      <th>cluster_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>-1.33</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>-0.28</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>-0.99</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>-0.29</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.06</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>-1.66</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>-0.97</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>-0.32</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>1.02</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8</td>
      <td>0.68</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>-0.34</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5</td>
      <td>0.39</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>-1.69</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2</td>
      <td>0.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>7</td>
      <td>0.27</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>9</td>
      <td>1.36</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>8</td>
      <td>1.38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>7</td>
      <td>1.36</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7</td>
      <td>-0.34</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>9</td>
      <td>0.67</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10</td>
      <td>1.18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3</td>
      <td>-1.69</td>
      <td>3</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4</td>
      <td>1.04</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3</td>
      <td>-0.96</td>
      <td>3</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6</td>
      <td>1.03</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>9</td>
      <td>-0.99</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10</td>
      <td>0.37</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>9</td>
      <td>0.03</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3</td>
      <td>-1.36</td>
      <td>3</td>
    </tr>
    <tr>
      <th>29</th>
      <td>5</td>
      <td>0.73</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'],c = clusters_new['cluster_pred'],cmap = 'rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyality')
plt.show()
```


![png](output_27_0.png)


## Heatmap


```python
x_scaled
```




    array([[-0.93138063, -1.3318111 ],
           [-0.15523011, -0.28117124],
           [-0.54330537, -0.99160391],
           [ 0.23284516, -0.29117733],
           [-0.93138063,  1.05964534],
           [-2.09560642, -1.6620122 ],
           [ 1.39707095, -0.97159172],
           [ 0.62092042, -0.32119561],
           [ 0.62092042,  1.01962097],
           [ 0.62092042,  0.67941378],
           [ 1.39707095, -0.3412078 ],
           [-0.54330537,  0.38923705],
           [-0.54330537, -1.69203048],
           [-1.70753116,  0.66940768],
           [ 0.23284516,  0.26916393],
           [ 1.00899568,  1.35982816],
           [ 0.62092042,  1.37984035],
           [ 0.23284516,  1.35982816],
           [ 0.23284516, -0.3412078 ],
           [ 1.00899568,  0.66940768],
           [ 1.39707095,  1.17971847],
           [-1.31945589, -1.69203048],
           [-0.93138063,  1.03963316],
           [-1.31945589, -0.96158562],
           [-0.15523011,  1.02962706],
           [ 1.00899568, -0.99160391],
           [ 1.39707095,  0.36922486],
           [ 1.00899568,  0.02901767],
           [-1.31945589, -1.36182938],
           [-0.54330537,  0.72944425]])




```python
sns.clustermap(x_scaled,cmap='mako')
```




    <seaborn.matrix.ClusterGrid at 0x208fc99e080>




![png](output_30_1.png)



```python

```
