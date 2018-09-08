## Database

系统环境下python mysql connector的安装见[这篇文章](http://dev.mysql.com/doc/connector-python/en/connector-python-installation.html)。

anaconda下的python，使用`​`​`​conda install mysql-connector-python`。

### Connect

```python
import mysql.connector

def load_db_config():
  '''实际上这个函数可能从文件里读取用户配置'''
  return {'user':'root', 'password':'123456', 'host':'localhost', 'database':'employees'}

cnx = mysql.connector.connect(**load_db_config()) # 注意：不支持with语句
  # ======以下的例子都在这部分里=======#
cnx.close()
```

### Query

```python
  from datetime import date
  cursor = cnx.cursor(buffered=True) # one cursor can handle only one query
  query = ("SELECT first_name, last_name, hire_date FROM employees "
           "WHERE hire_date BETWEEN %s AND %s")
  hire_start = date(1999, 1, 1)
  hire_end = date(1999, 12, 31)

  cursor.execute(query, (hire_start, hire_end)) # is not identical to python %

  for (first_name, last_name, hire_date) in cursor:
    print("{}, {} was hired on {:%d %b %Y}".format(
      last_name, first_name, hire_date))
  cursor.close()
```

### Modify

```python
  from datetime import date, datetime, timedelta
  cursor = cnx.cursor(buffered=True)
  # Insert new employee
  add_employee = ("INSERT INTO employees "
                  "(first_name, last_name, hire_date, gender, birth_date) "
                  "VALUES (%s, %s, %s, %s, %s)")
  data_employee = ('Geert', 'Vanderkelen', tomorrow, 'M', date(1977, 6, 14))
  cursor.execute(add_employee, data_employee)
  emp_no = cursor.lastrowid # get the foreign key by the way

  # Insert salary information
  add_salary = ("INSERT INTO salaries "
                "(emp_no, salary, from_date, to_date) "
                "VALUES (%(emp_no)s, %(salary)s, %(from_date)s, %(to_date)s)")
  tomorrow = datetime.now().date() + timedelta(days=1)
  data_salary = {
    'emp_no': emp_no,
    'salary': 50000,
    'from_date': tomorrow,
    'to_date': date(9999, 1, 1),
  }
  cursor.execute(add_salary, data_salary)

  # Make sure data is committed to the database
  cnx.commit()
  cursor.close()
```

### Creating tables

[click here](http://dev.mysql.com/doc/connector-python/en/connector-python-example-ddl.html).

## Web API
```sh
pip install flask flask-cors
```

Code for web app:

```python
from flask import Flask, request, jsonify, abort
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/app_name', methods=['POST']) # arbitrary app name is OK
def app_name():
    if not request.json or 'content' not in request.json:
        abort(400)
    content = request.json['content'] # example to get 'content' from input json
    pass # do something here
    return jsonify({k1: v1, k2: v2}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0')
```

test code from terminal:

```sh
curl -H "Content-Type: application/json" -X POST http://localhost:5000/app_name -d '{"key":"value"}'
```

python code to use the API:

```sh
import requests
headers = {'Content-Type': 'application/json'}
r = requests.post('http://localhost:5000/app_name', json={"key": "value"}, headers=headers)
answer = r.json() # a python object containing response
```

## Crawler

### 资源链接

便捷处理URL的包：[beautifier](https://github.com/sachinvettithanam/beautifier)

修复混乱的Unicode编码：[ftfy](https://github.com/LuminosoInsight/python-ftfy) [unidecode](https://github.com/avian2/unidecode)

主要用的python库：[requests](http://docs.python-requests.org/en/master/user/quickstart/)

主要用的辅助软件：postman

[The Ultimate Wget Guide](http://www.thegeekstuff.com/2009/09/the-ultimate-wget-download-guide-with-15-awesome-examples)

可视化快速配置方案：

[集搜客软件](http://www.gooseeker.com/pro/product.html) [编程接口](http://www.gooseeker.com/land/python.html)

[import.io](https://www.import.io/builder/)

### 知识

爬虫针对一个网页需要保存的信息：

* url（可以访问该页最新的内容）
* 访问时间
* 网页标题
* html（网页快照）
* 网页正文（可选，看业务需求）
* 业务需要的结构化数据
* 业务需要的半结构化数据（如果是用关系型数据库存则这个可以统一存成JSON字符串）

### 爬虫代码案例

```python
import os
import re
import time
import traceback
from tqdm import tqdm
import requests
import bs4
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
```

```python
try:
    r = requests.get(url, timeout=10)
    # 如果网页编码不是utf-8需要像下面这样设置
    # r.encoding = 'gb2312'
    html = r.text
    soup = BeautifulSoup(html, 'lxml')
except requests.exceptions.Timeout as e:
    # 连接超时，考虑放入等待重新连接的对列
except requests.exceptions.RequestException as e:
    # 其他异常，预示着对url本身连接错误，大多数情况下应该放弃这个url
```

```python
links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
title = soup.find(class_='core_title_txt').text
posts = [p.text for p in soup.find_all(id=re.compile("post_content_"))] # id="post_content_137"
next_page_url =
    [urljoin(url, a['href']) for a in soup.find_all('a', href=True) if a.text=='下一页'][0]
# 如果Chrome复制下来的selector是这样 body > div.mainWarp > div > div.main > div > div > ul.zsList
soup.body.find('div', class_='mainWarp').div.find('div', class_='main').div.div.find('ul', class_='zsList')
```

## Other tools

[Python Documentation](https://docs.python.org/3/)

[Python Standard Library](https://pymotw.com/3/)

便捷的管理时间的包：[Arrow](http://arrow.readthedocs.io/en/latest/)

其他扩展：[拼音](https://github.com/lxneng/xpinyin) [形式化计算](http://live.sympy.org/) [地图可视化](https://github.com/python-visualization/folium) [并行/分布式计算](http://dask.pydata.org/en/latest/) [繁简转换](https://github.com/BYVoid/OpenCC)

使用python操控其他命令行程序[pexpect](https://pexpect.readthedocs.io/en/stable/)

BeautifulSoup文档：[英文](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) [中文](http://beautifulsoup.readthedocs.io/zh_CN/latest/)

[WordCloud](http://amueller.github.io/word_cloud/index.html)

数据可视化：

* 浏览器交互可视化[Bokeh](http://bokeh.pydata.org/en/latest/)（封装d3.js）
* [echarts](http://echarts.baidu.com/) [rawgraphs](http://rawgraphs.io/) [orange](https://orange.biolab.si/#Orange-Features)

命令行打印彩色 [termcolor](https://github.com/hfeeki/termcolor) 命令行打印表格 [prettytable](https://github.com/vishvananda/prettytable)

分析时间开销：[https://github.com/rkern/line\_profiler](https://github.com/rkern/line_profiler)

分析空间开销：[https://pypi.python.org/pypi/memory\_profiler](https://pypi.python.org/pypi/memory_profiler)

发布python模块：[setuptools](https://setuptools.readthedocs.io/en/latest/setuptools.html)

在notebook中调用matlab函数：`​`%load\_ext oct2py.ipython

给他人推荐学习python的方式：

If this is the first time you are trying to use Python, there are many good Python tutorials on the Internet to get you started. Mark Pilgrim’s [Dive Into Python](http://www.diveintopython.net/) is one that I personally suggest. If this is the first time you ever try to use a programming language, A [Byte of Python](http://swaroopch.com/notes/Python) is even better. If you already have a stable programming background in other languages and you just want a quick overview of Python, [Learn Python in 10 minutes](http://www.poromenos.org/tutorials/python) is probably your best bet.

