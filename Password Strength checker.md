```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("C:\\Users\\91878\\Downloads\\archive\\data.csv", on_bad_lines='skip')
print(data.head())
```

          password  strength
    0     kzde5577         1
    1     kino3434         1
    2    visi7k1yr         1
    3     megzy123         1
    4  lamborghin1         1
    


```python
data = data.dropna()
data["strength"] = data["strength"].map({0: "Weak", 
                                         1: "Medium",
                                         2: "Strong"})
print(data.sample(5))
```

              password strength
    160101    g9o6975d   Medium
    488327  buothe1798   Medium
    295893  v6c4bi17ka   Medium
    596596     karan07     Weak
    213900   128828hpp   Medium
    


```python
data.describe()
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
      <th>password</th>
      <th>strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>669639</td>
      <td>669639</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>669639</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>fxx4pw4g</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>496801</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
      <th>password</th>
      <th>strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>kzde5577</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kino3434</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>visi7k1yr</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>megzy123</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>4</th>
      <td>lamborghin1</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>669635</th>
      <td>10redtux10</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>669636</th>
      <td>infrared1</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>669637</th>
      <td>184520socram</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>669638</th>
      <td>marken22a</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>669639</th>
      <td>fxx4pw4g</td>
      <td>Medium</td>
    </tr>
  </tbody>
</table>
<p>669639 rows × 2 columns</p>
</div>




```python
def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character
  
x = np.array(data["password"])
y = np.array(data["strength"])

tdif = TfidfVectorizer(tokenizer=word,token_pattern=None)
x = tdif.fit_transform(x)

print(f'x shape: {x.shape}, y shape: {y.shape}')
print(f'x type: {type(x)}, y type: {type(y)}')
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.05, 
                                                random_state=42)

```

    x shape: (669639, 153), y shape: (669639,)
    x type: <class 'scipy.sparse._csr.csr_matrix'>, y type: <class 'numpy.ndarray'>
    


```python
print(f'xtrain shape: {xtrain.shape}, ytrain shape: {ytrain.shape}')
print(f'xtest shape: {xtest.shape}, ytest shape: {ytest.shape}')

```

    xtrain shape: (636157, 153), ytrain shape: (636157,)
    xtest shape: (33482, 153), ytest shape: (33482,)
    


```python
model = RandomForestClassifier()
model.fit(xtrain, ytrain)
print(f'Accuracy: {model.score(xtest, ytest)}')
```

    Accuracy: 0.9564243474105489
    


```python
import getpass
user = getpass.getpass("Enter Password: ")
data = tdif.transform([user]).toarray()

output = model.predict(data)
print(output)
```

    Enter Password:  ········
    

    ['Medium']
    
