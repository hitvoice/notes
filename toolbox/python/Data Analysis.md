NumPy
-----

```python
import numpy as np
```

> Attention: When assigning a float value to an integer array, the value will be rounded without a warning!

### Creating NumPy Arrays

**from python list**

```python
np.array([1,2,3]) # 1-d
np.array([[1,2,3],[4,5,6],[7,8,9]]) # 2-d
np.asarray(arr) # convert python list to array, but pass np.array through without copy 
arr.tolist() # convert to (possibly nested) python list. Don't use `list(arr)`!
```

**from built-in method**

```python
np.arange(0,10) # [0,10)
np.arange(0,11,2) # array([0,  2,  4,  6,  8, 10])
np.linspace(2.0, 3.0, num=5) # [2., 2.25, 2.5, 2.75, 3.]
np.linspace(2.0, 3.0, num=5, endpoint=False) # [2., 2.2, 2.4, 2.6, 2.8]
np.zeros(3) # 3 x 1
np.empty_like(x) # matrix with the same dimensions as 'x' with all 0s.
np.ones((3,3)) # 3 x 3 noticed that a tuple should be used
np.full((2,2), 7) # array([[7,7],[7,7]])
np.tile(x,(2,1)) # stacking multiple copies
np.linspace(0,10,3) # array([  0.,   5.,  10.])
np.eye(4) # 4 x 4 identity matrix
np.random.seed(1234)
np.random.rand(2,2) # uniform [0,1)
np.random.randn(2,2) # standard normal
np.random.randint(1,100) # 1 random number in [1,100)
np.random.randint(1,100,(2,2)) # 2x2 random numbers in [1,100)
np.random.permutation(arr) # not in-place
np.random.choice(arr, 10) # random sampling
np.indices((3,3)) # array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]])
```

**from existing arrays**

```python
np.hstack([a,b,c]) # left-middle-right. attention: a vector is by default a row vector
np.vstack([a,b,c]) # up-middle-down
np.concatenate([a,b]) # === np.vstack [(3,4),(3,4)]=>(6,4)
np.concatenate([a,b],axis=1) # === np.hstack [(3,4),(3,4)]=>(3,8)
np.stack([a,b]) # [(3,4),(3,4)]=>(2,3,4)
np.stack([a,b], axis=1) # [(3,4),(3,4)]=>(3,2,4)
np.stack([a,b], axis=2) # [(3,4),(3,4)]=>(3,4,2)

np.diag(x) # get diagonal vector from a matrix, or generate diagonal matrix from a vector
```

### file IO

```python
assert filename.endswith('.npz')
# save to disk
np.savez(filename, A=arr1, B=arr2)
np.savetxt(filename, arr, fmt='%d,%.5f', header='id,value', comments='') 
# load from disk
loader = np.load(filename)
arr1 = loader['A']
arr2 = loader['B']
# efficient compressed IO
import bcolz
def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]
```

### Editing

```python
s = np.array([1,3,4])
s = np.insert(s, 1, 2) # return [1,2,3,4]
s = np.append(s, 5) # return [1,3,4,5]

s = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
s = np.delete(s, 1, axis=0) # [[1,2,3,4],[9,10,11,12]]
```

### Indexing

Similar to MATLAB indexing. Index using a np.array with type bool. (using python native array won’t work)

```python
a = np.array([[1,2], [3, 4], [5, 6]])
# WARNING: be careful when using a[a>2]=2 in functions, use `np.where` instead
a[a > 2] # [3 4 5 6]
a[(a<2) | (a>6)] # use & | , don't forget `()`
a[~np.isnan(a)]
```

Note that this fancy indexing method consumes tremendous memory. Be cautious when dealing with large arrays.

A slower but smaller alternative:

```python
import itertools
def mask_select(arr, mask):
    result = []
    for x in itertools.compress(arr.flatten(), mask.flatten()):
        result.append(x)
    return np.array(result)
```

### Attributes and Methods

```python
arr = np.arange(21)
# methods
arr.reshape(3,7)# == arr.reshape(3,-1) == arr.reshape(-1,7), vector is filled in each row
arr.flatten() # [[1,2], [3,4]] => [1, 2, 3, 4]
arr.flatten('F')  # [[1,2], [3,4]] => [1, 3, 2, 4]
arr.max()
arr.argmax()
arr.min()
arr.argmin()
arr.quantile(.75)
arr.clip(min=0.05,max=0.95) # clipping values to make them lie within [0.05, 0.95]
arr.astype(int) # convert element type, other types: np.float32 np.float64
arr.flatten() # shape: (1000,1) -> (1000,), usually following "reshape"
arr[arr.nonzero()] # get only the non-zero elements from an array

# attributes
arr.shape
arr.dtype # don't write to this value, use "astype"
arr.size # total number of elements === np.prod(arr.shape)
```

### Operations

**arithmetic**

+,-,\*,/,\*\* with array or scalar.

**universal**

```python
np.sum(arr) # sum up all elements，'axis=0' sum by column ⬆️⬇️, 'axis=1' sum by row ⬅️➡️
# usually use with `reshape`: np.sum(arr, axis=0).reshape((-1,1))
np.sqrt(arr)
np.exp(arr)
np.log(arr)
np.max(arr) # only one input array
np.maximum(arr1, arr2) # element-wise maximum, broadcasting supported
np.sin(arr)
np.dot(A,B) # matrix dot product, or A.dot(B)

np.partition(arr, 3)[3] # 4th-min
np.percentile(arr, [25,50,75], axis=0) # with interpolation

np.count_nonzero(arr)
np.isnan(arr)
np.unique(arr)

np.array_equal(arr1, arr2) # True if two arrays have the same shape and elements
np.allclose(arr1, arr2) # True if 2 arrays are nearly the same (due to numerical issues)

np.argwhere(arr < 0) # index of target elements. e.g. [[-1, 1],[1,-1]] => [[0,0], [1,1]]
np.where(arr>10, 10, arr) # a two-way mapping
np.where(arr, 1, 0) # binarize data
np.where(arr>10) # equvilant to np.argwhere(arr>10).T
np.apply_along_axis(func1d, axis, arr) # apply function taking 1d-arr as input along axis
np.vectorize(ufunc)(arr) # apply element-wise function (transformation) to array

np.einsum('xyz->zxy', arr) # swap axis (higher dimension transpose)
np.expand_dims(arr, axis=0) # shape: (3,4)->(1,3,4)
np.squeeze(arr) # or arr.squeeze(), shape: (1,3,4)->(3,4)
np.flip(arr, axis=0) # inverse the array
np.argsort(arr)
```

**matrix**

```python
A.dot(B)
A.T # transpose, taking transpose of a vector does nothing.
```

**broadcasting**

```python
# example: normalize each row 
# (notice: in sparse matrix this should be done by diag construction)
x = np.array([[1, 2, 7], [30, 20, 50], [.04, .03, .03]])
s = np.sum(x, axis=1).reshape((-1,1))
x = x/s
```

Pandas
------

```python
import pandas as pd
from numpy.random import randn
df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())
```

> Attention: when shuffling a DataFrame or doing any cross-row transformation, remember to use `df.reset_index(inplace=True)` to erase all the original ids.

### IO

```python
df.index.name = 'idx'
df.to_csv('example.csv', index=False) # header=False
df = pd.read_csv('example.csv')
df = pd.read_csv('example.csv', header=None, names=['col1', 'col2', 'col3'])
df = pd.read_csv(parse_dates=['date_col'], infer_datetime_format=True)
# datetime columns can be sliced as:
df[(df['date'] > '2016-6-1') & (df['date'] <= '2016-6-10')]
# when reading a file with unknown encoding, the easiest way to deal with it is open the file in sublime text, and save as UTF-8
# report and skip bad lines instead of throwing exception:
df = pd.read_csv('example.csv', error_bad_lines=False)

# dump to string
from io import StringIO
output = StringIO()
df.to_csv(output, index=False)
output = output.getvalue() # str
# parse from string
df = pd.read_csv(StringIO(text))

pd.read_excel('output.xlsx',sheetname='Sheet1') # no '_'
df.to_excel('output.xlsx',sheet_name='Sheet1',index=False)
# to multiple sheets
writer = pd.ExcelWriter('output.xlsx')
df1.to_excel(writer,'Sheet1')
df2.to_excel(writer,'Sheet2')
writer.save()

# print entire dataframe or series
print(df.to_string())

# convert to numpy array
df.as_matrix()
# columns not in df will not appear (unlike `.loc` which will result in new NaN columns)
df.as_matrix(columns=['X','Y']) 

```

Database IO：[SQLAlchemy](http://www.sqlalchemy.org)，[pandas.read\_sql](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql.html#pandas.read_sql)

Reading JSON files：[pd.read\_json](http://pandas.pydata.org/pandas-docs/version/0.19.2/generated/pandas.read_json.html)

### Series

```python
s = pd.Series([1,2,3,4],index = ['USA', 'Germany','USSR', 'Japan']) # or initialize with dict
s['USA']
s.to_frame(name='ColName') # convert to one-column DataFrame
# when applying the following functions to DataFrames, they apply on each column.
# they can also be applied to groupby objects. # df.groupby('colX').sum(), colX become index
s.mean()
s.std()
s.min()
s.max()
s.abs()
s.count()
s.cumsum()
s.describe()
s.isnull().sum()
s.notnull().sum()
s.unique()
s.nunique() # number of unique values
s.duplicated(keep='first') # or 'last', return duplicated items
s.value_counts() # sorted in descending order: s.value_counts().head(10)
s.value_counts(sort=False).sort_index() # sorted by index instead of counts
s.value_counts(normalize=True)
s.map({'female':0, 'male':1}) # not in-place, can be used in feature extraction
s.map(len)
s.agg(['mean', 'var']) # return a dataframe
tqdm.pandas();s.progress_map(func) # can see a progress bar
s.astype(str)
# set operations
np.intersect1d(s1.values, s2.values)
np.union1d(s1.values, s2.values)
np.setdiff1d(s1.values, s2.values)
np.setxor1d(s1.values, s2.values) # sorted, unique values that are in only one input array.
```

### DataFrames

Be careful with chained indexing (see [this](http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy)). If the warning is false positive, use `pd.options.mode.chained_assignment = None`​.

```python
df = pd.DataFrame(np.random.randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())
# selecting
df[['W','Z']] # select columns
df.filter(regex='sales_\d+') # select specific columns
df.loc['A'] # select rows by name of index, return row as Series object
df.iloc[2] # select rows by indexing numbers
df.loc[['A','B'], ['W','Y']] # select rows and columns (A, W) & (B, Y)
df = df.loc[:, ['W','Y','A','B']] # reorder columns
# `.ix` can be `loc` or `iloc`. do not use it for indexing to avoid unexpected results.

df[df['W']>0] # results in rows ABDE
df[(df['W']>0) & (df['Y'] > 1)] # & |
df[df.index < 100]
df.sample(10, random_state=101)

# editing
df['new'] = df['W'] + df['Y'] # create a new column
df.drop('new',axis=1) # remove a column, not in-place
df.drop('new',axis=1,inplace=True) # the same as `del df['new']`
df.drop('E',axis=0)
df = df.drop(df[df['W']<0].index) # equiv to `df = df[df['W']>=0]`

df.reset_index() # reset to default 0,1...n index
df['States'] = 'CA NY WY OR CO'.split()
df.set_index('States',inplace=True)
df.reindex(np.random.permutation(df.index)) # shuffle the DataFrame rows (not in-place)
df[~df.index.duplicated(keep='first')] # remove duplicated indices
df[~df.duplicated(['some_column'], keep='first')] # remove duplicated rows
df[df.duplicated('some_column', keep=False)].sort_values(by='some_column')#see duplicated rows
df.transpose()

df.rename(columns={'A':'a','W':'w'}, inplace=True) # rename columns
df.rename(index=lambda x:x+1, inplace=True) # rename indices

# iterate
for index, row in df.iterrows():
  print(row['A'], row['B'])
```

### operations

```python
df.head(5)
df.info()
df.columns
df.index
df.sort_values(by='X', inplace=True, ascending=False) 
df[['X','Y']].corr() # correlation matrix
df.apply(lambda x: x.max() - x.min()) # reduce to a Series
df.apply(np.abs) # apply element-wise. Use 'applymap' to force element-wise operation
df.apply(lambda x: x['X']-x['Y'], axis=1) # row-wise, passes a Series object, returns a Series
tqdm.pandas();df.progress_apply(func) # can see a progress bar
df.where(df>0, arr) # if condition is not satified(<0), replace by elements in array arr.
```

### special creation

```python
# cross table
pd.crosstab(df['X'], df['Y'], margins=True) # count row totals and column totals
pd.crosstab([['a','b']], [['d','e']], rownames=['typeX'], colnames=['typeY'])
# typeY  d  e
# typeX      
# a      1  0
# b      0  1

# one-hot (dummy) variable
pd.get_dummies(train['Sex'], drop_first=True) # `drop_first` use n-1 bits to encode n classes
pd.get_dummies(train['Sex'], prefix='Sex')
# pivot table
data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}
df = pd.DataFrame(data)
# NaN will appear in the pivot table where there're no value
df.pivot_table(values=['D'],index=['A', 'B'],columns=['C']) 
# the average will be taken when multiple values are mapped to one cell
df.pivot_table(values=['D'],index=['A'],columns=['C']) 

groupby = df.groupby('key')
for key, g_df in groupby:
  pass
group = groupby.get_group('key')
groups = groupby.groups # {key: df_row_index}
```

### handle missing data

```python
df.dropna() # remove rows containing missing data
df.dropna(subset=['X', 'Z']) # consider only these columns
df.dropna(axis=1) # drop columns
df.dropna(thresh=2, inplace=True) # keep rows with at least 2 non-NA values
df['X'].fillna(value=df['X'].mean())
```

### multiple frames

```python
pd.concat([df1,df2,df3])

# default inner, no NaNs. If there's multiple keys, values equal on each key will be kept.
pd.merge(left, right, on=['key1', 'key2']) 
# the result contains all keys, NaNs may appear in both sides
pd.merge(left, right, how='outer', on=['key1', 'key2']) 
# the result contains keys in the left, NaNs may appear in right side.
pd.merge(left, right, how='left', on=['key1', 'key2'])
# the result contains keys in the right, NaNs may appear in left side.
pd.merge(left, right, how='right', on=['key1', 'key2'])
# if the columns are named differently. Not Recommended. Do rename first.
pd.merge(left, right, left_on='key1', right_on='key2')

left.join(right) # base on index, default 'left'
left.join(right, how='outer')
```

## Scientific Computing

### Correlation between tow ranks

```python
from scipy.stats import spearmanr
print(spearmanr([1,2,3,4,5],[5,6,7,8,7]))
print(spearmanr([1,2,3,4,5],[3,1,5,4,2]))
```

### Rank each row of a matrix

```python
def rank_row(arr):
    idx = arr.argsort(axis=1)
    ranks = np.empty(arr.shape, int)
    ranks[np.mgrid[:arr.shape[0], :arr.shape[1]][0], idx] = np.arange(arr.shape[1])
    return ranks
```

sparse matrix
-------------

### specification

* data: non-zero entries in left-to-right top-to-bottom ("row-major") order.
* indptr: indptr[0]=0, indptr[i+1]-indptr[i] = number of non-zero entries in row i
* indices: column indices of non-zero entries

Example:
$$ 
\begin{pmatrix}
0&0&0&0\\\\
5&8&0&0\\\\
0&0&3&0\\\\
0&6&0&0\\\\
\end{pmatrix}
$$
```
   data  = [ 5 8 3 6 ]
   indptr = [ 0 0 2 3 4 ]
   indices = [ 0 1 2 1 ]
```
reference: [CSR Format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29)
### Code Usage

Doc: [scipy.sparse](https://docs.scipy.org/doc/scipy-0.18.1/reference/sparse.html) [csr\_matrix](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)

```python
import numpy
import scipy.sparse as sp

# construction
X = sp.lil_matrix((nrows,ncols))
# constructing...
X = X.tocsr()

# IO
import numpy as np
def save_sparse_csr(filename, array):
    assert filename.endswith('.npz')
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    assert filename.endswith('.npz')
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                          shape=loader['shape'])
# expand
sp.hstack((a,b)).tocsr() # default coo

# traverse
cx = X.tocoo()
for i, j, v in zip(cx.row, cx.col, cx.data):
    (i, j, v) # warning: built-in method are much much more faster than traversal.
```

### Entropy from data

```python
p_data = pd.Series(arr).value_counts()/len(arr) # calculates the probabilities
entropy = scipy.stats.entropy(p_data)
```

