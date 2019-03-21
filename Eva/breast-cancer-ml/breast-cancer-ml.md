
# Breast cancer classification challenge

## Part 0: All the packages

### Step 1: The essentials


```python
!pip install pandas
!pip install numpy
!pip install matplotlib
```

### Step 2: The "some things that not everyone has installed"


```python
!pip install json
!pip install sklearn
!pip install eli5
!pip install tensorflow
!pip install keras
```

### Step 3: Some news to read while waiting

#### Best overall result:
Area under ROC curve: 0.9408866995073892<br/>
Recall: 0.7931034482758621<br/>
Precision: 0.8518518518518519<br/>
    
#### Best recall and AUC:
Area under ROC curve: 0.9584017515051998<br/>
Recall: 0.8275862068965517<br/>
Precision: 0.7741935483870968

## Part 1: Data import and preparation
### Step 1: Read in JSON measurements


```python
import os
import json

data_dir = './RecruitmentData-JSON/'

# Columns to store records' fields
recs = {
    'ID': [],
    'Surgery': [],
    'Body_Temp': [],
    'Cancer': [],
    'Left_Temps': [],
    'Right_Temps': []
}

for record in os.listdir(data_dir):
    with open(data_dir + record) as rec:
        data = json.load(rec)
        
        for field in data.keys():
            recs[field].append(data[field])
```


```python
# Sanity check
print(recs['Left_Temps'][0]) # first left breast
```

    [32.35638888888889, 29.651944444444446, 30.430833333333325, 31.217222222222226, 30.373055555555553, 31.196944444444437, 31.41333333333333, 31.14861111111111, 31.811666666666667, 31.304166666666674, 32.124722222222225, 31.33555555555556, 32.05055555555556, 31.884444444444448, 32.095, 32.31666666666668, 32.41194444444445, 32.247499999999995, 32.16861111111112, 30.58388888888889, 29.692777777777774, 29.383611111111104, 29.539166666666663, 29.24277777777778, 29.102222222222224, 29.516111111111115, 31.6675, 32.780833333333334, 29.89527777777777, 29.663888888888884, 31.02472222222222, 29.009166666666673, 29.360833333333332, 30.449722222222217, 30.181388888888893, 31.531666666666677, 31.185000000000002, 31.385277777777773, 31.51416666666666, 31.01611111111111, 31.161944444444444, 32.221388888888896, 32.33916666666667, 33.01555555555556, 32.07083333333333, 31.189444444444444, 30.48972222222223, 29.904166666666658, 29.526111111111113, 29.751944444444447, 29.46444444444444, 29.52083333333334, 29.724444444444444, 28.025555555555563, 28.120555555555566, 28.43833333333333, 29.05083333333333, 29.373055555555556, 30.302499999999995, 30.151666666666674, 31.134444444444437, 31.064444444444444, 31.27833333333334, 31.084166666666672, 32.618333333333325, 32.07722222222222, 31.38694444444444, 30.61277777777777, 30.210555555555555, 30.00944444444444, 30.40583333333333, 30.43416666666667, 27.89444444444445, 27.960000000000004, 28.141111111111112, 28.60249999999999, 29.477499999999996, 29.79472222222222, 30.31416666666665, 30.866388888888896, 31.390833333333337, 31.427777777777774, 30.72333333333334, 30.901944444444432, 30.369722222222233, 30.341388888888883, 28.477777777777778, 28.26527777777777, 29.428333333333327, 30.04944444444445, 30.254166666666663, 30.46111111111111, 30.165000000000006, 30.64055555555556, 29.439166666666658, 30.599444444444448]


### Step 2: Put measurements in a dataframe and omit post-surgery cases


```python
import pandas as pd

dataset = pd.DataFrame(recs)
dataset.head()
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
      <th>ID</th>
      <th>Surgery</th>
      <th>Body_Temp</th>
      <th>Cancer</th>
      <th>Left_Temps</th>
      <th>Right_Temps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>115</td>
      <td>0</td>
      <td>35.8</td>
      <td>0</td>
      <td>[32.35638888888889, 29.651944444444446, 30.430...</td>
      <td>[30.349722222222226, 28.32555555555555, 28.115...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>154</td>
      <td>0</td>
      <td>35.6</td>
      <td>0</td>
      <td>[31.091388888888886, 29.169999999999998, 28.63...</td>
      <td>[28.53888888888889, 27.670833333333334, 28.504...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>[32.24555555555557, 30.27777777777778, 30.0636...</td>
      <td>[33.01444444444445, 30.63055555555555, 29.5402...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>98</td>
      <td>1</td>
      <td>36.1</td>
      <td>0</td>
      <td>[31.71194444444444, 28.13416666666667, 27.1297...</td>
      <td>[29.64055555555556, 28.330277777777777, 28.401...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>210</td>
      <td>0</td>
      <td>36.7</td>
      <td>1</td>
      <td>[34.371111111111105, 32.69611111111111, 31.474...</td>
      <td>[34.27194444444444, 31.854166666666668, 30.913...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Taking rows where the column value is no surgery
dataset = dataset.loc[dataset['Surgery'] == 0]
dataset = dataset.reset_index(drop=True) # reset indices
dataset.head()
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
      <th>ID</th>
      <th>Surgery</th>
      <th>Body_Temp</th>
      <th>Cancer</th>
      <th>Left_Temps</th>
      <th>Right_Temps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>115</td>
      <td>0</td>
      <td>35.8</td>
      <td>0</td>
      <td>[32.35638888888889, 29.651944444444446, 30.430...</td>
      <td>[30.349722222222226, 28.32555555555555, 28.115...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>154</td>
      <td>0</td>
      <td>35.6</td>
      <td>0</td>
      <td>[31.091388888888886, 29.169999999999998, 28.63...</td>
      <td>[28.53888888888889, 27.670833333333334, 28.504...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>[32.24555555555557, 30.27777777777778, 30.0636...</td>
      <td>[33.01444444444445, 30.63055555555555, 29.5402...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>210</td>
      <td>0</td>
      <td>36.7</td>
      <td>1</td>
      <td>[34.371111111111105, 32.69611111111111, 31.474...</td>
      <td>[34.27194444444444, 31.854166666666668, 30.913...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>[31.337500000000002, 29.987222222222222, 29.69...</td>
      <td>[31.66611111111111, 29.794999999999998, 29.053...</td>
    </tr>
  </tbody>
</table>
</div>



1. Rewrite left and right individual sensors' temperatures into separate columns


```python
import numpy as np

temps_to_cols = {
    'Left_Temps': None,
    'Right_Temps': None
}

col_names = {
    'Left_Temps': [],
    'Right_Temps': []
}


for side in ['Left_Temps', 'Right_Temps']:
    num_meas, num_sensors = len(dataset[side]), len(dataset[side][0])
    temps_to_cols[side] = np.array(
        [dataset[side][x] for x in range(num_meas)])
    col_names[side] = list(map(lambda idx: side[:side.find('T')].lower() + str(idx), range(num_sensors)))
```


```python
# Sanity check
pd.DataFrame(temps_to_cols['Left_Temps'], columns=col_names['Left_Temps']).head()
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
      <th>left_0</th>
      <th>left_1</th>
      <th>left_2</th>
      <th>left_3</th>
      <th>left_4</th>
      <th>left_5</th>
      <th>left_6</th>
      <th>left_7</th>
      <th>left_8</th>
      <th>left_9</th>
      <th>...</th>
      <th>left_86</th>
      <th>left_87</th>
      <th>left_88</th>
      <th>left_89</th>
      <th>left_90</th>
      <th>left_91</th>
      <th>left_92</th>
      <th>left_93</th>
      <th>left_94</th>
      <th>left_95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.356389</td>
      <td>29.651944</td>
      <td>30.430833</td>
      <td>31.217222</td>
      <td>30.373056</td>
      <td>31.196944</td>
      <td>31.413333</td>
      <td>31.148611</td>
      <td>31.811667</td>
      <td>31.304167</td>
      <td>...</td>
      <td>28.477778</td>
      <td>28.265278</td>
      <td>29.428333</td>
      <td>30.049444</td>
      <td>30.254167</td>
      <td>30.461111</td>
      <td>30.165000</td>
      <td>30.640556</td>
      <td>29.439167</td>
      <td>30.599444</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31.091389</td>
      <td>29.170000</td>
      <td>28.633611</td>
      <td>28.805278</td>
      <td>29.184444</td>
      <td>30.118056</td>
      <td>30.163889</td>
      <td>29.694722</td>
      <td>29.980556</td>
      <td>29.998889</td>
      <td>...</td>
      <td>30.279167</td>
      <td>28.741944</td>
      <td>29.433333</td>
      <td>29.310833</td>
      <td>29.321944</td>
      <td>29.999167</td>
      <td>29.675278</td>
      <td>29.688056</td>
      <td>30.292500</td>
      <td>29.269444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32.245556</td>
      <td>30.277778</td>
      <td>30.063611</td>
      <td>29.356667</td>
      <td>29.539722</td>
      <td>29.766944</td>
      <td>29.575278</td>
      <td>29.540000</td>
      <td>30.222222</td>
      <td>30.460556</td>
      <td>...</td>
      <td>27.842222</td>
      <td>28.418611</td>
      <td>28.835833</td>
      <td>28.660278</td>
      <td>28.528611</td>
      <td>28.615833</td>
      <td>28.235278</td>
      <td>28.087778</td>
      <td>27.993889</td>
      <td>28.285000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34.371111</td>
      <td>32.696111</td>
      <td>31.474722</td>
      <td>31.261389</td>
      <td>32.046111</td>
      <td>31.988889</td>
      <td>31.983056</td>
      <td>31.178333</td>
      <td>31.789444</td>
      <td>32.158611</td>
      <td>...</td>
      <td>31.933889</td>
      <td>31.724444</td>
      <td>32.073056</td>
      <td>31.918056</td>
      <td>32.127500</td>
      <td>31.090000</td>
      <td>31.993611</td>
      <td>31.632500</td>
      <td>31.661944</td>
      <td>31.725000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31.337500</td>
      <td>29.987222</td>
      <td>29.697500</td>
      <td>29.725278</td>
      <td>30.421389</td>
      <td>30.700278</td>
      <td>30.270278</td>
      <td>30.508611</td>
      <td>31.275278</td>
      <td>30.761111</td>
      <td>...</td>
      <td>30.780833</td>
      <td>28.801667</td>
      <td>29.107778</td>
      <td>31.408611</td>
      <td>30.601389</td>
      <td>30.557500</td>
      <td>31.263333</td>
      <td>30.330556</td>
      <td>30.298333</td>
      <td>30.925000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 96 columns</p>
</div>



2. Put back to the main dataframe


```python
for side in ['Left_Temps', 'Right_Temps']:
    temps_df = pd.DataFrame(temps_to_cols[side], columns=col_names[side])
    dataset = pd.concat([dataset, temps_df], axis=1) # bind dataframes column-wise
dataset.head()
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
      <th>ID</th>
      <th>Surgery</th>
      <th>Body_Temp</th>
      <th>Cancer</th>
      <th>Left_Temps</th>
      <th>Right_Temps</th>
      <th>left_0</th>
      <th>left_1</th>
      <th>left_2</th>
      <th>left_3</th>
      <th>...</th>
      <th>right_86</th>
      <th>right_87</th>
      <th>right_88</th>
      <th>right_89</th>
      <th>right_90</th>
      <th>right_91</th>
      <th>right_92</th>
      <th>right_93</th>
      <th>right_94</th>
      <th>right_95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>115</td>
      <td>0</td>
      <td>35.8</td>
      <td>0</td>
      <td>[32.35638888888889, 29.651944444444446, 30.430...</td>
      <td>[30.349722222222226, 28.32555555555555, 28.115...</td>
      <td>32.356389</td>
      <td>29.651944</td>
      <td>30.430833</td>
      <td>31.217222</td>
      <td>...</td>
      <td>29.157500</td>
      <td>29.332500</td>
      <td>28.830000</td>
      <td>29.890000</td>
      <td>31.011111</td>
      <td>31.836944</td>
      <td>30.625000</td>
      <td>30.421389</td>
      <td>28.832500</td>
      <td>31.108333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>154</td>
      <td>0</td>
      <td>35.6</td>
      <td>0</td>
      <td>[31.091388888888886, 29.169999999999998, 28.63...</td>
      <td>[28.53888888888889, 27.670833333333334, 28.504...</td>
      <td>31.091389</td>
      <td>29.170000</td>
      <td>28.633611</td>
      <td>28.805278</td>
      <td>...</td>
      <td>30.085000</td>
      <td>28.626111</td>
      <td>28.819722</td>
      <td>28.804167</td>
      <td>29.219444</td>
      <td>31.330833</td>
      <td>30.599444</td>
      <td>29.778056</td>
      <td>30.038889</td>
      <td>28.714722</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>[32.24555555555557, 30.27777777777778, 30.0636...</td>
      <td>[33.01444444444445, 30.63055555555555, 29.5402...</td>
      <td>32.245556</td>
      <td>30.277778</td>
      <td>30.063611</td>
      <td>29.356667</td>
      <td>...</td>
      <td>27.778889</td>
      <td>27.856389</td>
      <td>27.969444</td>
      <td>28.260556</td>
      <td>28.319444</td>
      <td>28.382778</td>
      <td>28.118889</td>
      <td>27.748333</td>
      <td>27.693889</td>
      <td>27.815278</td>
    </tr>
    <tr>
      <th>3</th>
      <td>210</td>
      <td>0</td>
      <td>36.7</td>
      <td>1</td>
      <td>[34.371111111111105, 32.69611111111111, 31.474...</td>
      <td>[34.27194444444444, 31.854166666666668, 30.913...</td>
      <td>34.371111</td>
      <td>32.696111</td>
      <td>31.474722</td>
      <td>31.261389</td>
      <td>...</td>
      <td>30.750000</td>
      <td>30.524167</td>
      <td>30.065833</td>
      <td>29.911389</td>
      <td>30.586667</td>
      <td>30.556944</td>
      <td>30.681111</td>
      <td>30.968333</td>
      <td>30.930278</td>
      <td>30.537500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>[31.337500000000002, 29.987222222222222, 29.69...</td>
      <td>[31.66611111111111, 29.794999999999998, 29.053...</td>
      <td>31.337500</td>
      <td>29.987222</td>
      <td>29.697500</td>
      <td>29.725278</td>
      <td>...</td>
      <td>29.514722</td>
      <td>28.994167</td>
      <td>29.059167</td>
      <td>29.678333</td>
      <td>30.199722</td>
      <td>30.055278</td>
      <td>29.761389</td>
      <td>29.666667</td>
      <td>29.100833</td>
      <td>29.674444</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 198 columns</p>
</div>




```python
print("Size of dataset before dropping:", len(dataset))
tmp = dataset.dropna(how='any', axis=0, inplace=False)
print("Size of dataset after dropping:", len(tmp))
```

    Size of dataset before dropping: 156
    Size of dataset after dropping: 155


Just one wrong entry, let's neglect it since we cannot restore information from NaN.


```python
dataset = tmp
dataset.reset_index(drop=True)
dataset.head()
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
      <th>ID</th>
      <th>Surgery</th>
      <th>Body_Temp</th>
      <th>Cancer</th>
      <th>Left_Temps</th>
      <th>Right_Temps</th>
      <th>left_0</th>
      <th>left_1</th>
      <th>left_2</th>
      <th>left_3</th>
      <th>...</th>
      <th>right_86</th>
      <th>right_87</th>
      <th>right_88</th>
      <th>right_89</th>
      <th>right_90</th>
      <th>right_91</th>
      <th>right_92</th>
      <th>right_93</th>
      <th>right_94</th>
      <th>right_95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>115</td>
      <td>0</td>
      <td>35.8</td>
      <td>0</td>
      <td>[32.35638888888889, 29.651944444444446, 30.430...</td>
      <td>[30.349722222222226, 28.32555555555555, 28.115...</td>
      <td>32.356389</td>
      <td>29.651944</td>
      <td>30.430833</td>
      <td>31.217222</td>
      <td>...</td>
      <td>29.157500</td>
      <td>29.332500</td>
      <td>28.830000</td>
      <td>29.890000</td>
      <td>31.011111</td>
      <td>31.836944</td>
      <td>30.625000</td>
      <td>30.421389</td>
      <td>28.832500</td>
      <td>31.108333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>154</td>
      <td>0</td>
      <td>35.6</td>
      <td>0</td>
      <td>[31.091388888888886, 29.169999999999998, 28.63...</td>
      <td>[28.53888888888889, 27.670833333333334, 28.504...</td>
      <td>31.091389</td>
      <td>29.170000</td>
      <td>28.633611</td>
      <td>28.805278</td>
      <td>...</td>
      <td>30.085000</td>
      <td>28.626111</td>
      <td>28.819722</td>
      <td>28.804167</td>
      <td>29.219444</td>
      <td>31.330833</td>
      <td>30.599444</td>
      <td>29.778056</td>
      <td>30.038889</td>
      <td>28.714722</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>[32.24555555555557, 30.27777777777778, 30.0636...</td>
      <td>[33.01444444444445, 30.63055555555555, 29.5402...</td>
      <td>32.245556</td>
      <td>30.277778</td>
      <td>30.063611</td>
      <td>29.356667</td>
      <td>...</td>
      <td>27.778889</td>
      <td>27.856389</td>
      <td>27.969444</td>
      <td>28.260556</td>
      <td>28.319444</td>
      <td>28.382778</td>
      <td>28.118889</td>
      <td>27.748333</td>
      <td>27.693889</td>
      <td>27.815278</td>
    </tr>
    <tr>
      <th>3</th>
      <td>210</td>
      <td>0</td>
      <td>36.7</td>
      <td>1</td>
      <td>[34.371111111111105, 32.69611111111111, 31.474...</td>
      <td>[34.27194444444444, 31.854166666666668, 30.913...</td>
      <td>34.371111</td>
      <td>32.696111</td>
      <td>31.474722</td>
      <td>31.261389</td>
      <td>...</td>
      <td>30.750000</td>
      <td>30.524167</td>
      <td>30.065833</td>
      <td>29.911389</td>
      <td>30.586667</td>
      <td>30.556944</td>
      <td>30.681111</td>
      <td>30.968333</td>
      <td>30.930278</td>
      <td>30.537500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>[31.337500000000002, 29.987222222222222, 29.69...</td>
      <td>[31.66611111111111, 29.794999999999998, 29.053...</td>
      <td>31.337500</td>
      <td>29.987222</td>
      <td>29.697500</td>
      <td>29.725278</td>
      <td>...</td>
      <td>29.514722</td>
      <td>28.994167</td>
      <td>29.059167</td>
      <td>29.678333</td>
      <td>30.199722</td>
      <td>30.055278</td>
      <td>29.761389</td>
      <td>29.666667</td>
      <td>29.100833</td>
      <td>29.674444</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 198 columns</p>
</div>



### Step 3: Form data and targets


```python
targets = dataset['Cancer']
targets.head()
```




    0    0
    1    0
    2    0
    3    1
    4    0
    Name: Cancer, dtype: int64




```python
data_cols = np.hstack([['Body_Temp'], dataset.columns.get_values()[6:]])
data = dataset[data_cols]
data.head()
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
      <th>Body_Temp</th>
      <th>left_0</th>
      <th>left_1</th>
      <th>left_2</th>
      <th>left_3</th>
      <th>left_4</th>
      <th>left_5</th>
      <th>left_6</th>
      <th>left_7</th>
      <th>left_8</th>
      <th>...</th>
      <th>right_86</th>
      <th>right_87</th>
      <th>right_88</th>
      <th>right_89</th>
      <th>right_90</th>
      <th>right_91</th>
      <th>right_92</th>
      <th>right_93</th>
      <th>right_94</th>
      <th>right_95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>35.8</td>
      <td>32.356389</td>
      <td>29.651944</td>
      <td>30.430833</td>
      <td>31.217222</td>
      <td>30.373056</td>
      <td>31.196944</td>
      <td>31.413333</td>
      <td>31.148611</td>
      <td>31.811667</td>
      <td>...</td>
      <td>29.157500</td>
      <td>29.332500</td>
      <td>28.830000</td>
      <td>29.890000</td>
      <td>31.011111</td>
      <td>31.836944</td>
      <td>30.625000</td>
      <td>30.421389</td>
      <td>28.832500</td>
      <td>31.108333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35.6</td>
      <td>31.091389</td>
      <td>29.170000</td>
      <td>28.633611</td>
      <td>28.805278</td>
      <td>29.184444</td>
      <td>30.118056</td>
      <td>30.163889</td>
      <td>29.694722</td>
      <td>29.980556</td>
      <td>...</td>
      <td>30.085000</td>
      <td>28.626111</td>
      <td>28.819722</td>
      <td>28.804167</td>
      <td>29.219444</td>
      <td>31.330833</td>
      <td>30.599444</td>
      <td>29.778056</td>
      <td>30.038889</td>
      <td>28.714722</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35.0</td>
      <td>32.245556</td>
      <td>30.277778</td>
      <td>30.063611</td>
      <td>29.356667</td>
      <td>29.539722</td>
      <td>29.766944</td>
      <td>29.575278</td>
      <td>29.540000</td>
      <td>30.222222</td>
      <td>...</td>
      <td>27.778889</td>
      <td>27.856389</td>
      <td>27.969444</td>
      <td>28.260556</td>
      <td>28.319444</td>
      <td>28.382778</td>
      <td>28.118889</td>
      <td>27.748333</td>
      <td>27.693889</td>
      <td>27.815278</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36.7</td>
      <td>34.371111</td>
      <td>32.696111</td>
      <td>31.474722</td>
      <td>31.261389</td>
      <td>32.046111</td>
      <td>31.988889</td>
      <td>31.983056</td>
      <td>31.178333</td>
      <td>31.789444</td>
      <td>...</td>
      <td>30.750000</td>
      <td>30.524167</td>
      <td>30.065833</td>
      <td>29.911389</td>
      <td>30.586667</td>
      <td>30.556944</td>
      <td>30.681111</td>
      <td>30.968333</td>
      <td>30.930278</td>
      <td>30.537500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>31.337500</td>
      <td>29.987222</td>
      <td>29.697500</td>
      <td>29.725278</td>
      <td>30.421389</td>
      <td>30.700278</td>
      <td>30.270278</td>
      <td>30.508611</td>
      <td>31.275278</td>
      <td>...</td>
      <td>29.514722</td>
      <td>28.994167</td>
      <td>29.059167</td>
      <td>29.678333</td>
      <td>30.199722</td>
      <td>30.055278</td>
      <td>29.761389</td>
      <td>29.666667</td>
      <td>29.100833</td>
      <td>29.674444</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 193 columns</p>
</div>



## Part 2: Exploratory data analysis
### Step 1: Some basic stats
1. Cancer dataset balance


```python
from matplotlib import pyplot as plt

plt.figure(figsize=(15, 6), dpi=300)
plt.bar(x=['No cancer', 'Cancer'], height=[sum(targets == 0), sum(targets == 1)], color=["blue", "red"])
plt.show()
```


![png](output_26_0.png)


Quite a large imbalance of data, so we have to be looking into recall, precision, and AUC-ROC.

2. Distribution of body temperatures


```python
bd_temps = list(data['Body_Temp'])

plt.figure(figsize=(15, 6), dpi=300)
plt.hist(bd_temps, density=True, bins=20)
plt.xlabel('Temperature (°C)')
plt.ylabel('Distributional density')
plt.show()
```


![png](output_29_0.png)


3. Average measurements on each sensor


```python
left_avgs = np.mean([data.iloc[idx][1:97] for idx in range(len(data))], axis=0) # column-wise mean
right_avgs = np.mean([data.iloc[idx][97:] for idx in range(len(data))], axis=0) # column-wise mean

plt.figure(figsize=(15, 6), dpi=300)
plt.subplot(2, 1, 1)
plt.bar(range(96), height=left_avgs, width=0.8)
plt.xticks(np.arange(0, 100, 5))
plt.ylabel('Temperature (°C)')

plt.subplot(2, 1, 2)
plt.bar(range(96), height=right_avgs, width=0.8)
plt.xticks(np.arange(0, 100, 5))
plt.xlabel('Sensor enumeration')
plt.ylabel('Temperature (°C)')

plt.show()
```


![png](output_31_0.png)


Should be pretty similar, since the sensors are located symmetrically on both cups!

### Step 2: Point temperatures of no cancer and cancer breasts

1. Let's first compare how the temperatures distributions look like in sample normal breasts and sample breasts with cancer.


```python
'''
Custom enumerator to make indices for graph positions
'''
def my_enum(xs, start=0, step=1):
    for x in xs:
        yield (start, x)
        start += step

'''
Temperature distribution grapher for each cup
'''
def graph_rl_temps(row_idx, data=data, targets=targets, show_bd_temp=True):
    plt.figure(figsize=(15, 6), dpi=300)
    num_rows = len(row_idx)
    
    for plot_id, row_id in my_enum(row_idx, start=1, step=2):
        rec = data.iloc[row_id]
        cancer_status = 'Cancer' if list(targets)[row_id] == 1 else 'No cancer'
        
        num_sensors_per_cup = int(len(rec[1:]) / 2)
        last_row = (plot_id == num_rows * 2 - 1)
        
        # Left cup measurements
        plt.subplot(num_rows, 2, plot_id)
        plt.bar(range(96), height=rec[1:1+num_sensors_per_cup], width=1)
        if show_bd_temp:
            bd_temp_ln, = plt.plot([0, 96], [rec[0], rec[0]], color="red", label='Body temp')
        
        plt.yticks(np.arange(0, 40, 5))
        plt.xticks(np.arange(0, 97, 10), rotation='vertical')
        plt.ylabel(cancer_status)
        if last_row:
            plt.xlabel('Left breast')
            
        # Right cup measurements
        plt.subplot(num_rows, 2, plot_id + 1)
        plt.bar(range(96), height=rec[1+num_sensors_per_cup:], width=1)
        if show_bd_temp:
            plt.plot([0, 96], [rec[0], rec[0]], color="red")
        
        plt.yticks(np.arange(0, 40, 5))
        plt.xticks(np.arange(0, 97, 10), rotation='vertical')
        if last_row:
            plt.xlabel('Right breast')
            
    if show_bd_temp:
        plt.legend(handles=[bd_temp_ln], loc='lower center')
    plt.show()
```


```python
graph_rl_temps([2, 3])
```


![png](output_36_0.png)


We can observe that in normal breasts, the sensors show a "wavy" fashion as different parts of the breast have different temperatures depending on the distance from body and exposure to body heat. Observational studies show that only 2-5% of breast cancers affect both breasts, so in most cases, we should see weird things happening in one of the breasts.

We can see that in the person with breast cancer, the left breast temperature measurements behave a bit "too flat" in some places. This can indicate that because of tumor obtaining blood and heat, the temperature near the tumor is pretty much even and similar to the tumor temperature. Of course, this line of reasoning can also be subject to confirmation bias, so I will go more in depth.

2. It would be nice to know in which breast the cancer tumor is, but even if we do not know, when we average out the measurements in cancer left or right breasts, if our hypothesis is correct, we should get a generally flatter profile of the temperature distribution.


```python
data_nocancer = data.loc[np.nonzero(targets == 0)]
data_cancer = data.loc[np.nonzero(targets == 1)]

avg_nocancer = np.mean(data_nocancer, axis=0)
avg_cancer = np.mean(data_cancer, axis=0)

avg_comp_df = pd.DataFrame(np.vstack([avg_nocancer, avg_cancer]))
avg_comp_df      
```

    /usr/local/lib/python3.6/site-packages/ipykernel_launcher.py:1: FutureWarning: 
    Passing list-likes to .loc or [] with any missing label will raise
    KeyError in the future, you can use .reindex() as an alternative.
    
    See the documentation here:
    https://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike
      """Entry point for launching an IPython kernel.





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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>183</th>
      <th>184</th>
      <th>185</th>
      <th>186</th>
      <th>187</th>
      <th>188</th>
      <th>189</th>
      <th>190</th>
      <th>191</th>
      <th>192</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>35.407200</td>
      <td>31.883387</td>
      <td>30.420836</td>
      <td>30.146149</td>
      <td>30.202138</td>
      <td>30.502342</td>
      <td>30.810491</td>
      <td>30.856298</td>
      <td>31.058171</td>
      <td>31.134871</td>
      <td>...</td>
      <td>29.547209</td>
      <td>29.333464</td>
      <td>29.538884</td>
      <td>29.947493</td>
      <td>30.296838</td>
      <td>30.606780</td>
      <td>30.292969</td>
      <td>30.045007</td>
      <td>29.646442</td>
      <td>29.978920</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35.396552</td>
      <td>32.235833</td>
      <td>30.947088</td>
      <td>30.557232</td>
      <td>30.641513</td>
      <td>30.815623</td>
      <td>31.098477</td>
      <td>31.225000</td>
      <td>31.461772</td>
      <td>31.617807</td>
      <td>...</td>
      <td>30.696571</td>
      <td>30.564080</td>
      <td>30.700757</td>
      <td>30.911897</td>
      <td>31.090920</td>
      <td>31.322749</td>
      <td>31.111552</td>
      <td>30.938190</td>
      <td>30.831561</td>
      <td>30.873199</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 193 columns</p>
</div>




```python
graph_rl_temps([0, 1], data=avg_comp_df, targets=[0, 1], show_bd_temp=False)
```


![png](output_40_0.png)


Unclear what is going on here. The challenge is probably that different people have tumors at diffent places, so considering that our dataset is pretty small, the "hot" parts pretty much evenly distribute out without showing obvious patterns. Sad.

## Part 3: Validation scheme

### Step 1: LOOCV for lighter models


```python
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import recall_score, precision_score, roc_auc_score, roc_curve

def loocv_preds(model, X, y, threshold=0):
    loocv = LeaveOneOut()
    loocv.get_n_splits(X)
    
    # Since we're doing LOOCV, calculating scores on the spot is not possible
    dcsn_fn = [] 
    preds = []
    
    for train_idx, test_idx in loocv.split(X):
        trgt = np.array(y) # target value
        model.fit(X.iloc[train_idx], trgt[train_idx])
        
        dist_from_bound = model.decision_function(X.iloc[test_idx])
        dcsn_fn.append(dist_from_bound)
        preds.append(int(dist_from_bound >= threshold))
        
    return dcsn_fn, preds
```

### Step 2: K-fold for neural networks

For larger neural networks, LOOCV will take forever to validate, so I'm creating this. I make sure to reset model weights each time I change training and validation sets, so that we don't have indirect information flow and so that the network would not suspect of anything in learned in the fold before.


```python
from sklearn.model_selection import KFold

def kfold_preds(model, X, y, threshold=0.5, epochs=25, n_splits=7):
    kf = KFold(n_splits=n_splits)
    dcsn_fn = [] 
    preds = []
    
    for train_idx, test_idx in kf.split(X): 
        # Reset weights and reset model 
        model.reset_states()
        model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
        
        
        trgt = np.array(y) # target value
        model.fit(X.iloc[train_idx], trgt[train_idx], epochs=epochs, verbose=0)
        dcsn = model.predict(X.iloc[test_idx])
        
        for elem in dcsn:
            dcsn_fn.append(elem)
            preds.append(elem >= threshold)
        
    return dcsn_fn, preds
```

### Step 3: Metrics


```python
def auc_roc(y, dcsn_fn):
    aucroc = roc_auc_score(y, dcsn_fn)
    
    fpr, tpr, _ = roc_curve(y, dcsn_fn, drop_intermediate=False)
    
    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.2f)' % aucroc)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive Rate')
    plt.ylabel('True positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    
    return aucroc

def rec_prec(y, preds):
    rec_sc = recall_score(y, preds)
    prec_sc = precision_score(y, preds)
    return rec_sc, prec_sc
```

## Part 4: Separability evaluation

### Step 1: Normalize the data


```python
means = data.mean(axis=0)
stds = data.std(axis=0)

norm_data = (data - means) / (stds + 1e-16)
norm_data.head()
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
      <th>Body_Temp</th>
      <th>left_0</th>
      <th>left_1</th>
      <th>left_2</th>
      <th>left_3</th>
      <th>left_4</th>
      <th>left_5</th>
      <th>left_6</th>
      <th>left_7</th>
      <th>left_8</th>
      <th>...</th>
      <th>right_86</th>
      <th>right_87</th>
      <th>right_88</th>
      <th>right_89</th>
      <th>right_90</th>
      <th>right_91</th>
      <th>right_92</th>
      <th>right_93</th>
      <th>right_94</th>
      <th>right_95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.514501</td>
      <td>0.249376</td>
      <td>-0.587489</td>
      <td>0.130107</td>
      <td>0.694777</td>
      <td>-0.155391</td>
      <td>0.232957</td>
      <td>0.353118</td>
      <td>0.001181</td>
      <td>0.442206</td>
      <td>...</td>
      <td>-0.366109</td>
      <td>-0.144017</td>
      <td>-0.578505</td>
      <td>-0.162093</td>
      <td>0.350101</td>
      <td>0.702438</td>
      <td>0.102760</td>
      <td>0.119525</td>
      <td>-0.622229</td>
      <td>0.571741</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.253888</td>
      <td>-0.537865</td>
      <td>-0.907441</td>
      <td>-1.125057</td>
      <td>-1.143731</td>
      <td>-1.047553</td>
      <td>-0.562022</td>
      <td>-0.579386</td>
      <td>-1.129996</td>
      <td>-0.986758</td>
      <td>...</td>
      <td>0.185001</td>
      <td>-0.560620</td>
      <td>-0.584804</td>
      <td>-0.861448</td>
      <td>-0.778419</td>
      <td>0.374918</td>
      <td>0.087201</td>
      <td>-0.266336</td>
      <td>0.093685</td>
      <td>-0.864793</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.527952</td>
      <td>0.180402</td>
      <td>-0.172013</td>
      <td>-0.126357</td>
      <td>-0.723434</td>
      <td>-0.780884</td>
      <td>-0.820738</td>
      <td>-1.018686</td>
      <td>-1.250376</td>
      <td>-0.798166</td>
      <td>...</td>
      <td>-1.185264</td>
      <td>-1.014574</td>
      <td>-1.105899</td>
      <td>-1.211573</td>
      <td>-1.345303</td>
      <td>-1.532859</td>
      <td>-1.423015</td>
      <td>-1.483731</td>
      <td>-1.297922</td>
      <td>-1.404598</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.687261</td>
      <td>1.503189</td>
      <td>1.433461</td>
      <td>0.859150</td>
      <td>0.728443</td>
      <td>1.100390</td>
      <td>0.816500</td>
      <td>0.778321</td>
      <td>0.024306</td>
      <td>0.424864</td>
      <td>...</td>
      <td>0.580137</td>
      <td>0.558786</td>
      <td>0.178878</td>
      <td>-0.148317</td>
      <td>0.082756</td>
      <td>-0.125889</td>
      <td>0.136922</td>
      <td>0.447573</td>
      <td>0.622667</td>
      <td>0.229153</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.527952</td>
      <td>-0.384704</td>
      <td>-0.364906</td>
      <td>-0.382046</td>
      <td>-0.442459</td>
      <td>-0.119113</td>
      <td>-0.133012</td>
      <td>-0.499984</td>
      <td>-0.496762</td>
      <td>0.023618</td>
      <td>...</td>
      <td>-0.153851</td>
      <td>-0.343554</td>
      <td>-0.438060</td>
      <td>-0.298422</td>
      <td>-0.160969</td>
      <td>-0.450533</td>
      <td>-0.423025</td>
      <td>-0.333146</td>
      <td>-0.462990</td>
      <td>-0.288813</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 193 columns</p>
</div>



### Step 2: PCA with two eigenvectors for visualization


```python
from sklearn import decomposition

n_components = 2
pca = decomposition.PCA(n_components=n_components, svd_solver='randomized', whiten=True)
pca.fit(norm_data)
print("Explained variance of components:", pca.explained_variance_ratio_) # how much information is retained
```

    Explained variance of components: [0.69780441 0.04143069]


Seems like 2 is a reasonable amount since eigenvectors that come after these will only be able to explain a tiny bit of variance (less than 4%), so we are retaining a good amount of information. This makes sense because when there is no tumor, the temperatures tend to behave correlatedly (e.g., if the body gets hotter, point A and point B on the cup will both capture slightly higher temperatures, just of varying degree of temperature increase). On the other hand, when there is tumor, areas around the tumor heat up just with slightly varying speeds.

1. Calculating back the single-value decomposition of original vectors to plot


```python
data_svd = pca.transform(norm_data)
data_svd[0] # check
```




    array([-0.0316851 ,  1.50810304])



2. Plotting to see have a decision boundary overview


```python
nocancer_idx = np.where(targets == 0)
cancer_idx = np.where(targets == 1)

plt.figure(figsize=(12, 6), dpi=300)
plt.scatter(*zip(*data_svd[cancer_idx]), color="red", label='Cancer')
plt.scatter(*zip(*data_svd[nocancer_idx]), color="blue",label='No cancer')
plt.xlabel('Principal component 1 (variance explained: {0}%)'.format(int(pca.explained_variance_ratio_[0] * 100)))
plt.ylabel('Principal component 2 (variance explained: {0}%)'.format(int(pca.explained_variance_ratio_[1] * 100)))
plt.legend(loc="upper right")
plt.show()
```


![png](output_59_0.png)


We can see that there are some "deep" overlaps of red dots inside the cluster of blue dots, and some blue dots in the cluster of red dots, thus, linear separation will create many false positives and negatives. Most likely, we care more about flagging the red dots inside the blue cluster at a cost of the blue dots in the red cluster. It's better to have a good recall score because the cost of in-hospital check-up is lower than the cost of life. 

3. Let's check out what the principal components found out about the "interesting" spots


```python
plt.figure(figsize=(15, 6), dpi=300)
plt.subplot(1, 2, 1)
plt.hist(pca.components_[0], bins=30)
plt.xticks(np.arange(0.035, 0.085, 0.01))
plt.xlabel('Variance contribution in component 1 (weight)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(pca.components_[1], bins=30)
plt.xlabel('Variance contribution in component 2 (weight)')
plt.show()
```


![png](output_62_0.png)


We can see that for PC1 with 69.78% variance explained, most of the weights are in the maximum range of 0.075, meaning that we have no particularly "outstanding" feature. It mostly makes sense because tumors can appear anywhere, so it is hard to observe a pin-point of "Hey, sensor 5 has higher temperature, and most of the time it is the one that detects cancer." Maybe we have to do some manipulations with combination of features to come up with separation.

## Part 5: Learning models

### Step 1: Naive linear separation


```python
from sklearn import svm

clf = svm.SVC(kernel='linear', gamma='scale', probability=True)
```


```python
dcsn_fn, preds = loocv_preds(clf, norm_data, targets)
print("Area under ROC curve:", auc_roc(targets, dcsn_fn))
```


![png](output_67_0.png)


    Area under ROC curve: 0.8924466338259442



```python
rec, prec = rec_prec(list(targets), np.array(preds).flatten())
print("Recall:", rec)
print("Precision:", prec)
```

    Recall: 0.5517241379310345
    Precision: 0.64


Pretty low recall because we were not able to flag cases when it is actually cancer.
1. Let's try playing around with the threshold


```python
dcsn_fn, preds = loocv_preds(clf, norm_data, targets, threshold=-0.9)
rec, prec = rec_prec(list(targets), np.array(preds).flatten())
print("Recall:", rec)
print("Precision:", prec)
```

    Recall: 0.8275862068965517
    Precision: 0.47058823529411764


27% gain in recall and 17% drop in precision. Not bad, considering that we would like to prioritize recall in this case.
2. We can play around a bit more to find the optimal threshold


```python
# better to use genetic algorithm to find convergence
# and having more data would be super helpful!

dcsn_fn, preds = loocv_preds(clf, norm_data, targets, threshold=-0.4)
rec, prec = rec_prec(list(targets), np.array(preds).flatten())
print("Recall:", rec)
print("Precision:", prec)
```

    Recall: 0.7931034482758621
    Precision: 0.6216216216216216


Relative to last measurement: 3% drop in recall and 15% gain in precision. Compared to no threshold: 24% gain in recall and 2% drop in precision.

3. Let's look at which features had a more impact


```python
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    norm_data, targets, test_size=0.2, random_state=42)

# Compute feature importances for any black-box estimator by
# measuring how score decreases when a feature is not available
clf.fit(X_train, y_train)
perm = PermutationImportance(clf, random_state=1).fit(X_test, y_test)
eli5.show_weights(clf, feature_names=list(norm_data.columns), top=50)
```





    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>



    

    

    

    

    

    


    

    

    

    
        

    

        
            
                
                
    
        <p style="margin-bottom: 0.5em; margin-top: 0em">
            <b>
    
        y=1
    
</b>

top features
        </p>
    
    <table class="eli5-weights"
           style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
        <thead>
        <tr style="border: none;">
            
                <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature weights. Note that weights do not account for feature value scales, so if feature values have different scales, features with highest weights might not be the most important.">
                    Weight<sup>?</sup>
                </th>
            
            <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
            
        </tr>
        </thead>
        <tbody>
        
            <tr style="background-color: hsl(120, 100.00%, 90.82%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.445
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_71
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 93.55%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.269
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_74
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 93.69%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.261
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_9
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 94.18%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.233
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_29
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 94.21%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.231
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_80
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 94.34%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.223
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_41
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 94.37%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.222
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_46
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 94.61%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.208
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_28
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 94.73%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.202
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_80
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 94.81%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.197
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_51
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 94.85%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.195
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_78
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 94.90%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.193
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_40
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 94.91%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.192
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_25
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.00%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.187
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_64
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.00%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.187
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_59
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.22%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.176
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_87
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.22%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.176
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_87
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.30%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.171
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_34
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.34%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.169
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_81
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.43%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.165
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_74
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.45%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.164
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_54
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.47%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.162
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_70
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.49%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.162
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_21
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.59%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.156
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_14
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.64%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.154
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_61
    </td>
    
</tr>
        
            <tr style="background-color: hsl(120, 100.00%, 95.65%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        +0.153
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_33
    </td>
    
</tr>
        
        
            <tr style="background-color: hsl(120, 100.00%, 95.65%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 82 more positive &hellip;</i>
                </td>
            </tr>
        

        
            <tr style="background-color: hsl(0, 100.00%, 95.64%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 62 more negative &hellip;</i>
                </td>
            </tr>
        
        
            <tr style="background-color: hsl(0, 100.00%, 95.64%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.154
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_6
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 95.63%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.154
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_76
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 95.61%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.155
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_59
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 95.55%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.159
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_70
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 95.46%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.163
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_86
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 95.46%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.163
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_42
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 95.42%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.165
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_62
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 95.30%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.171
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_2
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 95.27%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.173
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_68
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 95.05%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.184
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_13
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 94.87%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.194
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_7
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 94.85%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.195
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_36
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 94.85%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.195
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_53
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 94.85%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.195
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_71
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 94.85%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.195
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_27
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 94.62%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.208
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_32
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 94.59%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.209
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_36
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 94.28%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.227
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_79
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 94.16%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.233
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        right_2
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 94.06%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.239
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_1
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 93.01%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.302
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        Body_Temp
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 92.61%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.327
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_82
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 91.91%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -0.372
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        left_40
    </td>
    
</tr>
        
            <tr style="background-color: hsl(0, 100.00%, 80.00%); border: none;">
    <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
        -1.356
    </td>
    <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
        &lt;BIAS&gt;
    </td>
    
</tr>
        

        </tbody>
    </table>

            
        

        



    

    

    

    


    

    

    

    

    

    


    

    

    

    

    

    







### Step 2: Bring in the kernels!

1. Radial-basis function (RBF) kernel


```python
clf2 = svm.SVC(kernel='rbf', gamma='auto')
dcsn_fn, preds = loocv_preds(clf2, norm_data, targets, threshold=0)
print("Area under ROC curve:", auc_roc(targets, dcsn_fn))
```


![png](output_78_0.png)


    Area under ROC curve: 0.8732895457033388



```python
dcsn_fn, preds = loocv_preds(clf2, norm_data, targets, threshold=-0.55)
rec, prec = rec_prec(list(targets), np.array(preds).flatten())
print("Recall:", rec)
print("Precision:", prec)
```

    Recall: 0.7931034482758621
    Precision: 0.6216216216216216


Same LOOCV performance as the linear kernel :)

2. Polynomial kernel


```python
clf2 = svm.SVC(kernel='poly', degree=3, coef0=1, gamma='auto')
dcsn_fn, preds = loocv_preds(clf2, norm_data, targets, threshold=0)
print("Area under ROC curve:", auc_roc(targets, dcsn_fn))
```


![png](output_81_0.png)


    Area under ROC curve: 0.8524904214559387



```python
dcsn_fn, preds = loocv_preds(clf2, norm_data, targets, threshold=-0.75)
rec, prec = rec_prec(list(targets), np.array(preds).flatten())
print("Recall:", rec)
print("Precision:", prec)
```

    Recall: 0.8275862068965517
    Precision: 0.46153846153846156


I have tried to vary the degree of polynomial and 3 seemed to work best for degrees 2-10. I tried to vary the threshold to find a good balance for recall and precision, but the recall drops significantly when I move the threshold closer to zero (less negative). Seems like it is hard to project our data to a dimension that would make sense without overfitting (e.g., number of dimensions = number of data points).

### Step 3: Time for neural nets

1. Experiment #1 with a toy network


```python
import keras
from keras.layers import Dense
np.random.seed(1337) # for reproducible results
from keras.models import Sequential
```


```python
# Building a multi-layer perceptron model
from keras.optimizers import Adam

adam = Adam(0.01)

nn_model = Sequential()
nn_model.add(Dense(8, input_shape=(193,), activation='relu'))
nn_model.add(Dense(8, input_shape=(8,), activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))
```


```python
dcsn_fn, preds = kfold_preds(nn_model, norm_data, targets, threshold=0.4, epochs=15)
```


```python
print("Area under ROC curve:", auc_roc(targets, dcsn_fn))
```


![png](output_89_0.png)


    Area under ROC curve: 0.9584017515051998



```python
rec, prec = rec_prec(list(targets), np.array(preds).flatten())
print("Recall:", rec)
print("Precision:", prec)
```

    Recall: 0.8275862068965517
    Precision: 0.7741935483870968


Not too shaby, the best result so far. 

2. Experiment #2 with a bit more serious network. Gotta look out for overfitting because out dataset is small.


```python
adam = Adam(0.01)

nn_model_2 = Sequential()
nn_model_2.add(Dense(16, input_shape=(193,), activation='relu'))
nn_model_2.add(Dense(16, input_shape=(16,), activation='relu'))
nn_model_2.add(Dense(16, input_shape=(16,), activation='relu'))
nn_model_2.add(Dense(1, activation='sigmoid'))
```


```python
dcsn_fn, preds = kfold_preds(nn_model_2, norm_data, targets, threshold=0.5, epochs=15)
print("Area under ROC curve:", auc_roc(targets, dcsn_fn))
```


![png](output_93_0.png)


    Area under ROC curve: 0.9408866995073892



```python
rec, prec = rec_prec(list(targets), np.array(preds).flatten())
print("Recall:", rec)
print("Precision:", prec)
```

    Recall: 0.7931034482758621
    Precision: 0.8518518518518519


Best cumulative result (although a bit worse on recall than the previous model due to elevated threshold).

## Part 6: Alternative approach to classification
So far, we have been utilizing all the features to come up with a decision boundary. However, as I mentioned before, it is very rare (2-5%) that someone has breast cancer on both sides. Thus, using sensor information from both sides might add more noise to the decision function. In the training dataset, the cancer in left or right breasts can be and should be evenly distributed, the right temperatures will also have some weights.

**Example**: there's a tumor in the left breast, but since we use both breasts to determine cancer, the right one would be "cooler" and more normally "temperaturized." Then, when we put together in a dot product, we can get "overwhelmed" by the right breast, for example, some abnormal patterns in the left breast might be averaged out by the weights in the right breast.

### Step 1: Classify which breast is more likely to have cancer
We'll see which breast temperature distribution is furthest from the averaged temperature distribution of corresponding healthy breasts. Whichever breast that is furthest from its respective mean, it will be chosen as the "likely." The other one we will put to the mean healthy breast value to not affect during classification.


```python

```

### Step 2: Classify the breast with higher likelihood of cancer


```python

```

## Part 7: Works-to-be-done
In **Part 5**, I used the undergrad student descent to figure our the optimal threshold (e.g. at 0.5 the recall was too low, at 0.3 the precision was too low, go to the middle). The hyperparameter selection could be done automated with grid or random selection in a pre-determined range, and then the range would be modified to be more precise. Techniques like genetic algorithm could also be used to avoid local minima.

For the neural network, if I had more time, I would play around with network depth to see at which point it will start overfitting and settle with that architecture. To counterplay overfitting, I would do data augmentation (oversampling) (via small wiggle around existing datapoints) especially on the cancer dataset to get more defined boundaries for this class. Dropout would also help the problem of overfitting on training dataset.

In this part, I have not looked into representing the sensor measurements as an image array. I would imagine that representing that way, we would avoid the correlated features (nearby sensors) better and find an assessment of temperature landscape as a group of sensors and not only individual sensors multiplied by some weights and added up together (losing information about locality).

Also, I think Part 6 might help boost a couple points in recall since we're "exposing" the irregularities more by not letting the non-affected breast confound the dot-product of measurements.

Last but not least, let me know if you have any questions!
