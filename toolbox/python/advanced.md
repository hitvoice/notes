- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html?showone=TODO_Comments)
- [Example Google Style Python Docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

## Utilities
### progress
```python
from tqdm import tqdm
for i in tqdm(iterable):
  time.sleep(0.01)
# manual
with tqdm(total=100) as pbar:
  for i in range(10):
    pbar.update(10)
    time.sleep(0.5)
# annotation
t = trange(100)
for i in t:
    t.set_description('GEN %i' % i)
    t.set_postfix(loss=random(), gen=randint(1,999), str='h', lst=[1, 2])
    time.sleep(0.1)
# GEN 97:  98%|█████████▊| 98/100 [00:10<00:00,  9.61it/s, gen=260, loss=0.749, lst=[1, 2], str=h]
# nested progress bars
for i in trange(10, desc='1st loop'):
    for j in trange(5, desc='2nd loop', leave=False):
        for k in trange(100, desc='3nd loop'):
            sleep(0.01)
# parallel progress bar:
from multiprocessing import Pool
with Pool(n_process, initializer=init) as p:
    result = list(tqdm(p.imap(func, arr, chunksize=256), total=len(arr), desc='parallel'))
```
### log
```python
import logging
# simple logging
logging.basicConfig(format='%(asctime)s %(message)s',
    level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

# print to screen and file
log = logging.getLogger(__name__)
if not log.handlers: # avoid adding handlers again
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler('main.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO) # WARN
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)
```
### argparse
```python
import argparse

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(
    description='Some description.'
)
parser.register('type', 'bool', str2bool)
parser.add_argument('-s', '--source_dir', default=source_dir)
parser.add_argument('--ratio', default=0.9, type=float)
parser.add_argument('-d', '--debug', action='store_true',  
                    help='use debug mode')
parser.add_argument('--cuda', type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available())
model = parser.add_argument_group('Model Architecture')
model.add_argument('--model-type', type=str, default='rnn')
args = parser.parse_args()
print(args.source_dir, args.cuda)
print(vars(args)) 
# use empty str: --para=

# simple parsing (use only in very simple case)
if len(sys.argv) == 3:
    arg_1, arg_2 = sys.argv[1], sys.argv[2] # sys.argv[0] is python filename

```
### time
```python
import time
import arrow
# sleep until 3 a.m. tomorrow
now = arrow.utcnow().to('local')
target = now.shift(days=1).replace(hour=3,minute=0,second=0) 
time.sleep((target-now).total_seconds())
```
### debugging
```python 
import pdb
pdb.set_trace() # add this line to create a breakpoint
# commands in debug mode：https://docs.python.org/3/library/pdb.html
# frequent commands：
s(tep)
n(ext)
c(ontinue)
p [expression] # Evaluate the expression in the current context and print its value.
pp [expression] # for pretty print
whatis [expression] # Print the type of the expression.
interact # enter interact mode (Ctrl+D to exit)
cl(ear) # clear all breakpoints
q(uit) # stop the debugger and program
```

## os and persistence
### os
```python
import os
basedir = os.getcwd()

os.makedirs('dirname', exist_ok=True) # == mkdir -p dirname
os.listdir() # ls
dirnames = [name for name in os.listdir() if os.path.isdir(name)] 
filenames = [name for name in os.listdir() if name[-4:] == '.jpg']
from glob import glob # https://docs.python.org/3/library/glob.html
glob('*.jpg') 
glob('**/*.csv',recursive=True) # .csv in current and all sub folders
glob('**/',recursive=True) # all folders

os.path.exists('filename')
os.path.join('parent','1st','2nd') # parent/1st/2nd
os.path.dirname('pipeline/data/1.txt') # pipeline/data
os.path.basename('pipeline/data/1.txt') # 1.txt
os.path.splitext('pipeline/data/1.txt') # ('pipline/data/1', '.txt')
os.path.basename('pipeline/data/') # empty ''
os.path.abspath('data/1.txt') # get absolute path

os.rename(src, dst) # can be used to move files (full path is needed)
os.remove(file) # remove a single file
from shutil import rmtree
rmtree(directory) # rm -r directory

from shutil import copyfile
copyfile(src, dst) # full path

def slugify(value):
    """make valid filename: remove illegal chars, tranlate spaces to "-""""
    value = re.sub('[^\w\.\s\-_\(\)]', '', value).strip()
    return re.sub('[-\t\r\n\v\f]+', '-', value)
```
### use bash
```python
import subprocess
sp = subprocess.Popen('git log -1 | head -n 1', shell=True,
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = sp.communicate()
stdout, stderr = stdout.decode(), stderr.decode()
```
### dump python object
```python
# for speed and minmum space, use msgpack (pip install msgpack-python)
import msgpack
with open('data.msgpack', 'wb') as f:
    msgpack.dump(obj, f)
with open('data.msgpack', 'rb') as f:
    obj = msgpack.load(f, encoding='utf8')
# for readability, use JSON
# for pure numbers, use bcolz
# don't use pickle: http://www.benfrederickson.com/dont-pickle-your-data/
```
## data processing
### regex
```python
import re
# 如果没有匹配以下都返回None，可以直接用作判断
re.match(pattern, string) # match from the beginning
m = re.search(pattern, string) # match in any position
m.group() # empty or use 0: get the entire matched result; use group id (starting from 1) to fetch a group
m.groups() # return a tuple
m.start() # start position. empty or use 0: entire match; use group id to refer to a group
m.end()
ms = re.findall(pattern, string) # return a list
s = re.sub(pattern, replaced_by, s) 
# replace_by can be a str or function 
# if str, can specify group as \g<1>, \g<2>, ...
# if function, in: a match object https://docs.python.org/3/library/re.html#match-objects; out: str

re.split(' |\n', s) # return a list
re.escape(s) # treat s as a literal
regex = re.compile(pattern) # re has internal caches, so most of the time compiling is not needed
```
### random
```python
import random
random.seed(101) # same seed will behave differently in python 2 and 3
random.choice(['A','B','C'])
random.randint(0,100) # [0,100]
random.randrange(0, 101, 2) # even integer in [0,100]
random.random() # [0.0,1.0)
random.uniform(0.5, 0.8) # [0.5,0.8]
random.shuffle([1, 2, 3, 4, 5]) 
random.sample(range(100),  3) # without replacement (no duplicates)
random.sample(arr, len(arr)) # not in-place shuffle
random.choices([1,2,3,4,5], k=3) # with replacement
```
## Jupyter notebook
```python
files = !ls
!echo {files}
dirname='dir/'
%ls {dirname}
%timeit X.dot(Y)
%matplotlib inline
%%bash # 整个cell变成bash脚本。
# 注意这里的cd不会影响笔记本的路径，下一个cell还是原来路径，但是%cd会
# 如果开头就定义函数，最好空一行，否则会有bug程序会卡住
# 脚本不是逐行反馈结果，而是整个脚本执行完毕后一起显示输出，所以不能通过输出情况判断进度

%run -i 'prepro.py' # 执行脚本，并且共享变量空间

??some_function # 可以直接看到源码，或者按两次shift+TAB
from IPython.display import FileLink
FileLink('Path/to/file.csv') # 可以返回一个超链接，可以直接下载到本地
from IPython.display import display
display(df) # pretty print (multiple) tables, ...

# tqdm in notebook
from tqdm import tnrange, tqdm_notebook
for i in tnrange(10, desc='1st loop'): # 代替trange
    for j in tqdm_notebook(range(100), desc='2nd loop'): # 代替tqdm，慎用内层循环（不会消失）
        sleep(0.01)
```
