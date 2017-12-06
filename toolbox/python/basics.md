```python
#基础知识
#============================================================#
x = 1/2 #数学除法
x = 1//2 #依然做整数除法
r = 78 % 60
x, r = divmod(78, 60) # 1, 18
x = 2**3 #乘方
x = 0xAF #十六进制
x = 0o10 #八进制
#表达式跨行
x = 1 + 2 + \
    4 + 5
#长字符串（三个单/双引号,其中单/双引号就不需要转义）
string = '''This is a very long string.
It continues here.
And it's not over yet.
"Hello, world!"
Still here.'''
#原始字符串(最后一个字符不能是反斜线)
string = r'C:\Program Files\fnord\foo'
#unicode编码转换
code = ord('🌟') # 127775
assert '🌟' == chr(code)
hash(s) # 对字符串和浮点数做哈希；注意整数（除了-1）哈希还是自己
#还支持二进制、复数等数据类型
0b1000100 == 34 == int('0b100010', 2)
bin(34) == '0b1000100'
# 特殊的数
float('nan'), float('inf')
# 检查是否是nan
math.isnan(x)
np.isnan(x)
x != x
# 得到存储环境变量的字典
locals()

#数据结构
#============================================================#
#列表[a,b,c,...]
#索引x[i]
    #从左至右：0,1,2,...
    #从右至左：-1,-2,-3,...
    #两种索引方式可以混用
#分片
    #x[i1:i2]
        #i1:要提取的第一个元素的编号，缺省则从第一个元素起
        #i2:分片之后剩下部分的第一个元素编号，缺省则取到最后一个元素，超出长度范围不要紧会自动忽略！
    #x[i1:i2:step]
        #step步长为正：从左到右；为负：从右到左
    #x[::-1] 会将列表反转
    #逻辑不对则返回空列表
#连接[a1,b1,c1,...] + [a2,b2,c2,...] 产生一个新列表
#注意千万不要用sum(nested, [])连接多个列表，而应用双层for循环做flattern，否则会慢成狗
#生成含有重复元素的列表[a,b,c,...]*n，空列表：[None]*10，注意这个是shallow copy应只用于常量
#检查成员资格：in
#长度len(x);最大值max(x);最小值min(x)
#---------------
del x[i] #删除元素
y = x[:] #复制列表
#列表方法:
x.append(new_item) #在末尾追加元素
x.insert(index,new_item) #在列表指定的索引前插入新对象
x.extend(another_list) #用新列表扩展原有列表，追加在末尾
x.count(item) #统计item出现的次数
x.index(item) #找出第一个匹配项的索引，找不到则引发异常ValueError
x.pop(index) #移除一个元素并返回其值，缺省为最后一个元素
x.remove(item) #移除第一个匹配项，没有返回值
x.reverse() #将元素反向存放(inplace)
x.sort() #原位置排序（从小到大）
y=sorted(x) #得到排序副本，x不变
#x可以是任何一种可迭代对象
x.sort(key = key_function, reverse = bool_value) #使用指定函数生成的键值来排序，或指定反向排序
#flatten a nested list:
import itertools
itertools.chain('ABC', 'DEF') #--> A B C D E F
#或者:(以下的做法效率超级低)
a=[[1],[2],[3,4],[5,6],[7,8,9]]
sum(a,[]) # 第二个参数是初始值
#生成全排列：
list(itertools.permutations([1,2,3]))
#生成全部组合：
list(itertools.combinations([1,2,3], 2)) # [(1, 2), (1, 3), (2, 3)]
#============================================================#
#元组(a,b,c,...)
#一个值的元组：(value,)
# x = (1,); a, = x; # a == 1
#上述对列表的说明中“删除元素”之前的内容对于元组都可用，方法中count和index可用
#元组和列表的主要区别在于，元组是一种不可变序列
#============================================================#
#字符串“abc...”
#注意字符串都是不可变的，所以会改变内容的操作总是返回新串
#字符串格式化 "...%s..." % ('1st_value','2nd_value',...) 或
name = "Fred"
f"He said his name is {name}." # 'He said his name is Fred.'
width = 10
precision = 4
value = 12.34567
f"result: {value:{width}.{precision}}"  # 'result:      12.35'
x = int(1e7)
f'{x:_}' # '10_000_000'
#字符串方法
s.find('substring'[,int_start,int_end]) #返回子串最左端索引，找不到返回-1
element.join(_list) #在列表元素间插入元素形成新字符串
s.split(sep = sep_element) #将字符串分隔成列表，缺省则用空格、制表、换行符分割
import re
re.split(' |\n', s) #如果要用多种定界符分割，需要用正则表达式
[w for w in s.split(sep) if w.strip()] #可以去除分割结果中的空字符串
s.lower() #转换成小写
s.replace('origin','new') #替换所有匹配项，如果找不到则返回一样的字符串
s.strip('target') #去除字符串两侧的指定字符，缺省则为去掉空格
s.ljust(10) # 如果没到指定长度，填充空格到这么长；如果超长不会发生变化
s.startswith('prefix')
s.endswith('.txt')
#匹配最长子串
m = SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2))
# m.a m.b 左边和右边的起始位置 m.size 匹配长度
substr = s1[m.a : m.a+m.size]
#============================================================#
#集合{1,3,4,...}
s = set() #注意不能用{}初始化，否则会被当作dict
s.add(1)
s.update([1,3,4])
s.remove(3)
s1 <= s2 # 还有>=, |, &, -, ^(symmetric difference，并集去掉交集)
#============================================================#
#字典{a:1,b:2,c:3,...}
#键值可以为任何常量，如果需要多个量作为键，拼成tuple即可
#如果d是字典，那么d[new_key] = new_value 就可以自动增加新的项
#k in d 查找的是键，而不是值
#用字典格式化字符串 "...%(key)s..." % d 会用value进行置换，不能和用元组的格式化混用
#字典方法
d.clear()
d2 = d1.copy() #注意如果替换值，d1不受影响，但是如果原地修改了值，则两个字典都受影响，原因是尽管d2是新字典但是引用了同一处值
from copy import deepcopy;d2 = deepcopy(d1) #可以实现完全的深拷贝
dict.fromkeys([key1,key2,..],value) #生成一个初值全是value的指定键的新字典，如果value缺省则初值为None
d.get(key,'notation for None') #宽松的访问，如果找不到键返回第二个参数指定的值，缺省为None
d.pop(key) #返回值并移除这一项
d.pop(key,None) #安全地从字典中删除一个key
d.popitem() #随机地从字典中pop
d.setdefault(key,default_value) #键不存在时返回default_value并将key值设成default_value,如果键存在则返回原值并不改变原值，缺省为None
d1.update(d2) #用d2更新d1，新的项会添加，相同的键会覆盖其值
d.keys() #返回键
d.items() #返回包含(键，值)的列表
#更高效的迭代方法：
for k,v in d.items():
  print(k,v)

# 特殊的字典：计数器 Counter
from collections import Counter
counter = Counter(iterable)
counter.update(iterable)
# 直接迭代得到的是key，比如要得到按频率排序的list：
vocab = sorted(counter, key=counter.get, reverse=True)
# 也可以用这个方法得到排序：
counter.most_common() # 'lily' -> [('l', 2), ('i', 1), ('y', 1)]
# 其他基本操作和dict类似，两个counter可以直接做加减运算，counter.keys()可以做减法


# deque适合队列和栈
from collections import deque
q = deque([1,2,3])
q.append(1)
q.appendleft(3)
q.pop()
q.popleft()

# 优先级队列
import heapq
heapq.nlargest(10, s, key=str.lower)
heapq.nsmallest(100, arr)

from collections import namedtuple
Point = namedtuple('Point',['x','y'])
p = Point(1,2)
print(p.x, p.y)

# 判断是不是集合：list, dict, set, Series, DataFrame, np.array...，不包括str
import collections
tell = lambda o: isinstance(o, collections.Iterable) and not isinstance(o, str)

#============================================================#
# 树
class Tree:
    def __init__(self, cargo, left=None, right=None):
        self.cargo = cargo
        self.left  = left
        self.right = right

    def __str__(self):
        return str(self.cargo)

tree = Tree(1, Tree(2), Tree(3))

#语句
#============================================================#
#同时赋值
x,y,z = 1,2,3
x,y,z = (1,2,3)
key,value = d.popitem()
x,y = y,x
x,y,*r = (1,2,3,4,5) #数量不一致又没使用最后一行的格式时会引发异常
#链式赋值
x = y = function()
#增量赋值
x += 1
x *= 2
#语句块（完全靠缩进来区分语句块，别忘了if，for和while最后的分号）
#若行末存在换行符\ 则下一行默认从头开始，缩进会被认为是加入空格，并且如何缩进对语句块没有任何影响
#============================================================#
#条件
if statement1:
    block1;
    Still block1;
elif statement2:
    block2;
else:
    block3;
#条件中statement只要“有些东西”就会判断成True，"没有东西"(False,None,0,"",(),[],{})则为False
#字符串和其他序列也可以直接用>,<,==进行比较
#短路逻辑
name = input("Please enter your name: ") or '<unknown>' #第一个值为空则用第二个值赋值++
x = value1 if statement else value2 #相当于三元运算符
#断言
assert statement,'Explaination' #程序在不满足断言时会“准时”崩溃，也可以写成if...raise...
#循环
#while
name = ''
while not name.strip():
    name = input("Enter your name: ")
#for
for number in range(1,101,2): #range(n)=range(0,n,1)，只能接受int,如果需要float用np.arange
    print(number) #打印1~100中的奇数
for key in d:
    print(key,"-",d[key])
for key,value in d.items():
    print(key,"-",value)
#并行迭代
for name,age in zip(names,ages):
    print(name,"is",age,'years old.')
#逆运算为：
zipped = zip(names, ages)
names, ages = zip(*zipped)
a[:], b[:] = zip(*np.random.permutation(list(zip(a,b)))) # 打乱次序同时保留相对位置
#编号迭代
for index,string in enumerate(strings):
    if '和谐词' in string:
        strings[index] = '由于相关法律法规，您索引的内容已被屏蔽'
for i, x in enumerate(arr, 2):
  pass # i从2开始计数，x还是按顺序取
#手动迭代（通常用于寻找first occurrence）
def first_occurrence(logfile,regex):
  return next(line for line in logfile if regex.match(line), None)
#跳出循环
for girl in girls:
    if is_bitch(girl):
        print("Bitch is %s!!!" % girl)
        break
else: #如果没有通过break跳出则执行以下（对while而言也一样）
    print("Didn't find the bitch!")
#列表推导式
[x*x for x in range(5)] #注意range(5)是从0到4
#如果是嵌套for循环那么外层循环写在前面
[(x*2,y) for x in range(10) if x%3 == 0 for y in range(3) if y != 2]
#不止是推导列表，如反转词典：
{v: k for k, v in d.items()}
#Set comprehensions
{int(sqrt(x)) for x in range(30)} #"set([0, 1, 2, 3, 4, 5])"

#函数
#定义函数
def function_name(parameter1 = default_value1,parameter2,*rest_parameter，**rest_pairs):#rest_parameter会把剩余普通参数收集成元组，rest_pairs会把接下来“key=value”这样的参数收集成字典。同时传入的时候可以用`**dic`，传入的字典会被解开成为key=value的形式，这样参数设置就可以统一存放在字典里管理了
    '这行会被作为文档字符串存储。'
    #do something here
    return (result1,result2,...)
#查看文档字符串
function_name.__doc__
help(function_name)
#函数嵌套
def func1(para1):
    def func2(para2):#此func2对外不可见
        return#处理步骤中可以直接使用para1和para2
    return func2
#嵌套函数的调用
#方法1
func1(para1)(para2)
#方法2
func3 = func1(para1)
func3(para2-1)
func3(para2-2)
# 指定参数包装成新函数
from functools import partial
basetwo = partial(int, base=2)
basetwo('10010') # 18

#异常处理
#看一个案例
import traceback
while True:
    try:
        x = int(input('Enter the first number: '))
        y = int(input('Enter the second number: '))
        value = x/y
        print('x/y is ',value)
    except (ZeroDivisionError,TypeError) as e:
        print('Invalid input:',e)
        print('Please try again')
    else:#no exceptions
        break
    except (KeyboardInterrupt, SystemExit):
        # 单独处理人工中断，可以选择再抛出
        raise
    except Exception as e:
        #处理意料之外的异常
        print(e)
        traceback.print_exc()
    finally:#不论有无异常都会执行
        pass#实际中用来关闭文件或网络套接字
#自定义异常类:
class CustomException(Exception): pass
#在需要引发的地方:
raise CustomException

#类
class Superclass:
    def __init__(self,value = defaultValue):
        pass


class Subclass(Superclass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) #如果不能够完全覆写父类构造函数需要添加新内容那么要这样调用父类构造函数
        pass

    def __len__(self):
        return self.MyLength #len(x)时的返回值
    def __contains__(self, key): # 'xxx' in x时调用
        return key in self.members
    def checkIndex(key):
        #检查键值合法性的范例
        if not isinstance(key,(int,long)):raise TypeError
        if key<0: raise IndexError
    # 对于某些场合，读取时不需要强制检查，而是定义键缺失时返回的值
    def __missing__(self, key):
        return key
    def __getitem__(self,key):
        checkIndex(key)
        return self.MyValue #x[key]返回值
    def __setitem__(self,key,value):
        #设置对应key的值为value,仅可变对象可用
        checkIndex(key)
        pass
    def __delitem__(self,key):
        #删除需要移除的元素，仅可变对象可用
        checkIndex(key)
        pass
    def __str__(self):
        return "prettified strings"

    def __call__(self, arg):
        #在类的对象像函数一样被调用时调用
        return arg

    #自定义属性操作
    def __setattr__(self, name, value):
      #在使用self.xxx = value时触发，为了避免无尽递归，最后用如下方式把值存进去
      super().__setattr__(name, value)

    def __getattr__(self, name):
      # 找到时不会触发，传入找不到的名字时才触发；如果还是个非法名字，raise AttributeError
      return None

    #静态属性，按照SubClass.static_variable访问
    static_variable = []

    @staticmethod
    def smethod():
        #不需要和类的实例以及类本身交互的静态方法(如检查标志位)，可通过类的实例或类本身访问
        pass
    @classmethod
    def cmethod(cls):
        #不需要和类的实例交互但要和类本身交互的静态方法，传入参数为类本身，可通过类的实例或类本身访问，访问时不需传入参数
        pass

    #若实例为可迭代对象
    def __iter__(self):
        return self
    def __next__(self):#若迭代长度有限需要在类内设置终止条件，若长度无限则在调用时加入break
        self.index += 1
        if self.index > self.upperBound: raise StopIteration #有限长度迭代器可以用list(x)转换成列表
        return self.data[index]
    #调用时：for x in xs: ... (xs是该subclass的实例)x就是__next__返回的单个data
    #也可以用next(xs)逐个取出
    #注意迭代过程会不断累加，所以下一次迭代如果需要从头迭代则需要显式“归零”
    # 另一种写法：
    def __iter__(self):
        while self.index < self.upperBound:
          self.index += 1
          yield self.index
    # 这种写法可以用for x in ..，但如果用next（比如无限循环的情况）需要xs=iter(xs); next(xs)


#生成器
#生成器推导式
#如果有多个for,后面的for是子循环
((i+2)**2 for i in range(2,27))#产生可迭代对象(生成器),适用于大量数据(注意此时只能迭代)
[(i+2)**2 for i in range(2,27)]#会立刻实例化整个列表
[random.random() for _ in range(10)] #如果不需要用控制变量的值，可以用下划线代替，这个表达式在一个列表中生成10个随机数
#生成器推导式可以在当前圆括号下立即使用：
sum((i+2)**2 for i in range(2,27))
# 取第i个数字，比如取第6个数字：
next(itertools.islice(range(10), 5, None))
#定义生成器(函数)
def generate_ints(N):
  for i in range(N):
    yield i
gen = generate_ints(2)
next(gen) # 0
next(gen) # 1
next(gen) # 异常：StopIteration
gen.close() # 提前关闭生成器

# 可重置的生成器示例
def counter(maximum):
    i = 0
    while i < maximum:
        val = (yield i)
        # If value provided, change counter
        if val is not None:
            i = val
        else:
            i += 1
it = counter(10)
next(it) # 0
next(it) # 1
it.send(8) # 8， 重置生成器
next(it) # 9

# 例子：对树进行前序遍历
def inorder(t):
  if t:
    for x in inorder(t.left):
      yield x

    yield t.cargo

    for x in inorder(t.right):
      yield x

#lambda表达式
list(filter(lambda word: word[0]=='s', words)) # 选出以s开头的词
list(map(lambda x:x*2, [1,2,3]))
max(d.keys(), key=lambda x: d[x]) # get the max-valued key

#读写文本文件
with open(filename) as f:
	for line in f: # f.readlines()，两者读取进来都是带有'\n'结尾的，需要注意！
	  if line == '\n':
	    continue
		print(line, end='')
#多个文件
with open(fn1) as f1, open(fn2) as f2:
  pass
#如果遇到包含可忽略的编码错误的文件：
with open(filename, encoding='utf-8', errors='ignore') as f
#如果sublime text可以看见python读进来却是乱码，可以尝试手工打开文件并选择save with encoding:utf-8

# pprint
from pprint import pprint, pformat
pprint(obj) # 输出到屏幕
with open(file) as fout:
  pprint(obj, stream=fout, compact=True) # 输出到文件
s = pformat(obj) # 转化成字符串

# 读取json
with open(file) as f:
    data = json.load(f)
# 读取以后json就变成了python原生格式了，因此可以动用原生方法进行探索
json_str = json.dumps(data) # data = json.loads(json_str)
with open(file, 'w') as f:
    json.dump(data, f) # ensure_ascii=False 可以写入中文明文,
                       # indent=2 可以pretty print， indent=0 只换行不缩进
```
