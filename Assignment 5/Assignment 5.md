
# Assignment 5
### Part 0: Libraries


```python
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
```


```python
from sklearn.mixture import GaussianMixture
```

### Part 1: Forming dataset


```python
transactions = pd.read_csv('anonymized.csv')
transactions.head()
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
      <th>Date</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25May2016</td>
      <td>54241.35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29May2017</td>
      <td>54008.83</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30Jun2017</td>
      <td>54008.82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>05Jan2017</td>
      <td>52704.37</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23Feb2017</td>
      <td>52704.36</td>
    </tr>
  </tbody>
</table>
</div>




```python
dates = list(map(lambda date: datetime.strptime(date, '%d%b%Y'), transactions['Date']))
```


```python
years = pd.Series(list(map(lambda date: date.year, dates)))
months = pd.Series(list(map(lambda date: date.month, dates)))
days = pd.Series(list(map(lambda date: date.day, dates)))
```


```python
years.name = 'Year'
months.name = 'Month'
days.name = 'Day'
transactions = pd.concat([transactions, years, months, days], axis=1)
```


```python
transactions.head()
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
      <th>Date</th>
      <th>Amount</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25May2016</td>
      <td>54241.35</td>
      <td>2016</td>
      <td>5</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29May2017</td>
      <td>54008.83</td>
      <td>2017</td>
      <td>5</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30Jun2017</td>
      <td>54008.82</td>
      <td>2017</td>
      <td>6</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>05Jan2017</td>
      <td>52704.37</td>
      <td>2017</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23Feb2017</td>
      <td>52704.36</td>
      <td>2017</td>
      <td>2</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>



### Part 2: Defining Gaussian mixture plotting function
Below I define a GMM with independent mixtures (no covariance) since we have only 1 dimension.


```python
def gmm_calc(weights, means, covs, rng=[0, 120]):
    x = np.linspace(rng[0], rng[1], 200)
    y = np.zeros((len(x), 1))
    
    for i in range(len(means)):
        exp = np.exp(-(x - means[i])**2 / (2*covs[i]**2)).reshape(-1, 1)
        y += weights[i] / np.sqrt(2*np.pi*covs[i]) * exp # update mixture
    
    return x, y

def gmm_plot(x, y, orig_distr):
    plt.figure(figsize=(10, 6), dpi=330)
    plt.hist(orig_distr, density=True) # rescale original freq to probabilities
    plt.plot(x, y)
    plt.show()
```

### Task 3: Density model for the number of transactions that occur in a single month


```python
task3 = transactions.sort_values(by=['Year', 'Month'])
trans_in_month = []
cur = -1
count = 0

for i in task3.index:
    if task3.loc[i, 'Month'] != cur:
        if cur != -1: # not first iteration
            trans_in_month.append(count)
        cur = task3.loc[i, 'Month']
        count = 0
    count += 1
```

For `n_components` from 1 to 3, the GMM seemed to be too "flat" and does not capture the bumps we have at $10$ and $58$. For `n_components` from 5 upwards, the GMM overfits and creates patterns that do not seem likely in real life (e.g., too many spikes). More formally, we can look at model likelihood and evaluate its tradeoff with the "weird curviness."

In reality, it looks like a mixture of 2 Gaussians with weights of about $0.3$ and $0.7$ would work well here because there are two humps but for some reason I was not able to fit properly when `n_components=2`.


```python
mm3 = GaussianMixture(n_components=4)
mm3.fit(np.array(trans_in_month).reshape(-1, 1))
print("Likelihood of model:", np.e**mm3.score(np.array(trans_in_month).reshape(-1, 1)))
```

    Likelihood of model: 0.012573904239640377



```python
x, y = gmm_calc(mm3.weights_, mm3.means_.flatten(), mm3.covariances_.flatten())
gmm_plot(x, y, trans_in_month)
```


![png](output_15_0.png)


### Task 4: Density model for the day in the month that a transaction will occur on


```python
task4 = transactions.sort_values(by=['Day'])
day_trans = [0] * 32 # 31 days (counting from 1)

for i in task4.index:
    day = task4.loc[i, 'Day']
    day_trans[day] += 1
```


```python
day_trans_exp = [] # expanded version
for i in range(len(day_trans)):
    for j in range(day_trans[i]):
        day_trans_exp.append(i)
```

Again, looks like there can be two Gaussians with weights about $0.4$ and $0.6$, but it `sklearn` doesn't fit nicely :( 4 components seem to almost fit and 5 components seem to describe to contours nicely.


```python
mm4 = GaussianMixture(n_components=5)
mm4.fit(np.array(day_trans_exp).reshape(-1, 1))
print("Likelihood of model:", np.e**mm4.score(np.array(day_trans_exp).reshape(-1, 1)))
```

    Likelihood of model: 0.031803070806674986



```python
x4, y4 = gmm_calc(mm4.weights_, mm4.means_.flatten(), mm4.covariances_.flatten(), [0, 35])
gmm_plot(x4, y4, day_trans_exp)
```


![png](output_21_0.png)


### Task 5: Density model for transaction size


```python
trans_size = list(transactions['Amount'])
trans_size[:5]
```




    [54241.35, 54008.83, 54008.82, 52704.37, 52704.36]



Highest likelihood achieved at `n_components=3`.


```python
mm5 = GaussianMixture(n_components=3)
mm5.fit(np.array(trans_size).reshape(-1, 1))
print("Likelihood of model:", np.e**mm5.score(np.array(trans_size).reshape(-1, 1)))
```

    Likelihood of model: 0.00020029542840949136



```python
x5, y5 = gmm_calc(mm5.weights_, mm5.means_.flatten(), mm5.covariances_.flatten(), [-40000, 40000])
gmm_plot(x5, y5, trans_size)
```


![png](output_26_0.png)


### Task 6: Sample a fictious transaction month
We first draw from the number of transactions in a month distribution


```python
num_of_trans = int(mm3.sample()[0][0][0])
```


```python
# storage
fake_days = []
fake_amts = []

# draw days in which a transaction happened. Within each day, draw the amount:
for i in range(num_of_trans):
    fake_days.append(int(mm4.sample()[0][0][0]))
    fake_amts.append(int(mm5.sample()[0][0][0]))
```


```python
fake_trans = pd.DataFrame({'Day': fake_days, 'Amount': fake_amts})
fake_trans.head()
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
      <th>Day</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24</td>
      <td>17389</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>-185</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>-152</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>-87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>-912</td>
    </tr>
  </tbody>
</table>
</div>



### Task 7: Explain flaws
1. On an individual level, we consider only amount of transaction without taking into account other external variables. For example, big businesses make large transactions, but there are not so many big businesses out there, thus, if another large transaction comes in, we might flag it as suspicious with our GMM even though it might not be, if it was made B2B.
2. Our dataset is only 2518 entries yet we try to extrapolate to individual months (I think there are about 12 * 4 of them), thus the distributions might not be representative of what's happening in reality. Thus, maybe we would flag transactions on day 15 with lowest probability, but it might be that in our dataset we didn't get to see day 15 as much compared to real life.
3. Another flaw would be that we don't really know what's actually suspicious. If fraudsters know about these data, they can blend in places with high probability. Then, we would lose track of them if we only look at single parameter at a time like here.


```python

```
