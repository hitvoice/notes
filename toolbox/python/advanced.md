- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html?showone=TODO_Comments)
- [Example Google Style Python Docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

```python
# 快速、压缩的序列化
# 带字符的用messagepack，纯数字的用bcolz (pip install msgpack-python)
import msgpack
with open('data.msgpack', 'wb') as f:
    msgpack.dump(obj, f)
with open('data.msgpack', 'rb') as f:
    obj = msgpack.load(f, encoding='utf8')
# 注意：不要使用pickle进行序列化
# 参考：http://www.benfrederickson.com/dont-pickle-your-data/

#计时/显示进度
# 最朴素（稳定）的计时
import time
now = time.time()
# do sth...
print("Finished in", time.time()-now , "sec")
# 如果已知长度会根据速度自动估算百分比，如果不知道长度(如文件流)则只会显示当前迭代次数
from tqdm import tqdm
for i in tqdm(iterable):
  time.sleep(0.01)
# 不规则的迭代手动更新：
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

# 并行计算
# for IO-intensive tasks, e.g. downloads, writing files to disks, UI
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    result = executor.map(func, arr)

# for CPU-intensive tasks. Due to python's GIL, one process cannot execute code simutanously
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    result = executor.map(func, arr, chunksize=64) # cannot use lambda here (which cannot be pickled)
# result is 'itertools.chain', can be converted to list

# parallel progress bar
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
def parmap(function, array):
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
        futures = [pool.submit(function, x) for x in array]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return out

# for functions that directly handle series, break up the input array and combine the results
def parmap_batch(func, arr, batch_size=64):
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    def flatten(batches):
            return [x for batch in batches for x in batch]
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        result = executor.map(func, batch(arr, n=batch_size))
    return flatten(result)

# Another library
from joblib import Parallel, delayed
res = Parallel(n_jobs=multiprocessing.cpu_count(), backend='threading') (delayed(func)(x) for x in arr)
res = Parallel(n_jobs=multiprocessing.cpu_count()) (delayed(func)(x) for x in arr)

import threading
tl = threading.local() # 如果要避免反复allocate大空间，可以用局部空间，具体要用到的话研究一下

# 重新加载模块
from importlib import reload # remove this line in Python 2
import chinese; reload(chinese)
from chinese import ChineseVectorizer

# 无法用pip安装（无法写进requirements.txt中）的模块
try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('Need to install pytorch: go to pytorch.org')

# 断点调试
import pdb
pdb.set_trace() #在需要断点的地方加入这么一行
# 进入断点模式后的命令：https://docs.python.org/3/library/pdb.html
# 常用命令：
s(tep)
n(ext)
c(ontinue)
p [expression] # Evaluate the expression in the current context and print its value.
pp [expression] # for pretty print
whatis [expression] # Print the type of the expression.
interact # enter interact mode (Ctrl+D to exit)
cl(ear) # clear all breakpoints
q(uit) # stop the debugger and program

# 日志
import logging
# 简单日志
logging.basicConfig(format='%(asctime)s %(message)s',
    level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)
# 分别输出到屏幕和文件
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

#随机数
import random
random.seed(101)
random.choice(['A','B','C'])
random.randint(0,100) # 产生[0,100]之间的随机数
random.randrange(0, 101, 2) # even integer in [0,100]
random.random() # [0.0,1.0)
random.uniform(0.5, 0.8) # [0.5,0.8]
random.shuffle([1, 2, 3, 4, 5]) # 注意python2和python3中尽管设置了同一个随机种子,shuffle的结果会不一样！
random.sample(range(100),  3) # 无放回采样，不会重复
random.sample(arr, len(arr)) # 非原位shuffle

# 正则表达式
import re
# 从头匹配，如果没有匹配以下都返回None，可以直接用作判断
re.match(pattern, string)
# 任意位置匹配一次
m = re.search(pattern, string)
m.group() # 或者传入组的编号（从1开始），不传或传0可以得到整个匹配
m.groups() # 返回一个tuple，每个组的匹配是其中一个元素
m.start() # 得到起始位置，传入组编号可以得到组的起始位置
m.end()
# 匹配多次
ms = re.findall(pattern, string) #返回包含字符串的列表
# 替换
# replace_by还可以是函数，传入的是match对象（参见https://docs.python.org/3/library/re.html#match-objects），返回的是纯字符串
s = re.sub(pattern, replaced_by, s)
# 分割
re.split(' |\n', s) # 返回列表
# 如果预先编译，调用其方法时省略上述函数第一个参数
# re模块内会自动缓存，所以大多数情况下不需要担心效率问题
regex = re.compile(pattern)
# 将任意字符串转换成正则表达式里的literal（或者有时候用r'...'也行）
re.escape(s)

# 时间相关
# 计算定时的时间差
import time
import arrow
now = arrow.utcnow().to('local')
target = now.shift(days=1).replace(hour=3,minute=0,second=0) # 定时在第二天凌晨3点
time.sleep((target-now).total_seconds())

# 系统相关
import os
basedir = os.getcwd()
%cd $dirname_variable

os.makedirs('dirname', exist_ok=True) # 支持递归创建，exist_ok默认为False
%mkdir -p parent/child # jupyter中的等价写法，在没有exist_ok的python2里尤其好用

os.listdir() # ls
dirnames = [name for name in os.listdir() if os.path.isdir(name)] #如果是深层文件也只有文件名
filenames = [name for name in os.listdir() if name[-4:] == '.jpg']
from glob import glob # https://docs.python.org/3/library/glob.html
glob('*.jpg') # 上面的等价写法，返回包含str的list
glob('**/*.csv',recursive=True) # 所有子文件夹里的csv文件,返回的是可以直接读取的相对地址
glob('**/',recursive=True) # 所有子文件夹，以/结尾限制了结果为文件夹

os.path.exists('filename')
os.path.join('parent','1st','2nd') # parent/1st/2nd
os.path.dirname('pipeline/data/1.txt') # pipeline/data
os.path.basename('pipeline/data/1.txt') # 1.txt
os.path.basename('pipeline/data/') # empty ''
os.path.abspath('data/1.txt') # get absolute path

os.rename(src, dst) # 也可以用来移动文件（需要完整文件路径＋文件名）
%mv srcfilr dstdir # 支持UNIX filename pattern
os.remove(file) # 移除单个文件
from shutil import rmtree
rmtree(directory) # 移除文件夹及其子内容

from shutil import copyfile
copyfile(src, dst) # 需要完整文件路径＋文件名
%cp src dst # 注意不要在内层循环里调用这个，否则会慢成狗...

def slugify(value):
    """将任意字符串转换成合法文件名：去除不合法字符、将除了空格的空白符转成-"""
    value = re.sub('[^\w\.\s\-_\(\)]', '', value).strip()
    return re.sub('[-\t\r\n\v\f]+', '-', value)

import time
time.sleep(5) # 5s

# 使用bash
import subprocess
sp = subprocess.Popen('git log -1 | head -n 1', shell=True,
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = sp.communicate()
stdout, stderr = stdout.decode(), stderr.decode()

# 命令行参数解析
import argparse
# 可以选择一种bool值的解析函数
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
    description='Train a Document Reader model.'
)
parser.register('type', 'bool', str2bool)
parser.add_argument('-s', '--source_dir', default=source_dir)
parser.add_argument('--train_ratio', default=0.9, type=float)
parser.add_argument('-d', '--debug', action='store_true',  #用了就会记为true
                    help='enter debug mode')

parser.add_argument('--cuda', type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available())
model = parser.add_argument_group('Model Architecture')
model.add_argument('--model-type', type=str, default='rnn')
args = parser.parse_args()
print(args.source_dir, args.cuda)
print(vars(args)) # convert Namespace to dict
# 如果需要传入空值，用--para=

# 简单解析（只在参数非常简单直白的时候用）
if len(sys.argv) == 3:
    arg_1, arg_2 = sys.argv[1], sys.argv[2] # 0是文件名

# 从标准输入读取（不需要prompt）
import sys
for line in sys.stdin:
  pass

# Jupyter notebook
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

# tqdm的notebook版本
from tqdm import tnrange, tqdm_notebook
for i in tnrange(10, desc='1st loop'): # 代替trange
    for j in tqdm_notebook(range(100), desc='2nd loop'): # 代替tqdm，慎用内层循环（不会消失）
        sleep(0.01)
```
