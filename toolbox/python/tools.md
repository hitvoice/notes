## Database

How to install python mysql connector for system default python: [link](http://dev.mysql.com/doc/connector-python/en/connector-python-installation.html).

For python in anaconda, use `conda install mysql-connector-python`.

### Connect

```python
import mysql.connector

def load_db_config():
  '''actually this function may acquire these information from a file or user input'''
  return {'user':'root', 'password':'123456', 'host':'localhost', 'database':'employees'}

cnx = mysql.connector.connect(**load_db_config()) # with statement is not supported
  # ======Examples below should be placed here=======#
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

### useful links
- handle URL: [beautifier](https://github.com/sachinvettithanam/beautifier)
- fix messy Unicode encoding: 
  - [ftfy](https://github.com/LuminosoInsight/python-ftfy) 
  - [unidecode](https://github.com/avian2/unidecode)
- send web requests: [requests](http://docs.python-requests.org/en/master/user/quickstart/)
- parse HTML: BeautifulSoup [En](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) [Zh](http://beautifulsoup.readthedocs.io/zh_CN/latest/)
- auto content extraction: [goose](https://github.com/grangier/python-goose)
- debugging tools：postman
- quick interactive solutions (some are not free)
  - [gooseeker](http://www.gooseeker.com/pro/product.html) ([python interface](http://www.gooseeker.com/land/python.html))
  - [import.io](https://www.import.io/builder/)
  
### basics

What is needed to be saved for a webpage:

* url (so you can update newer contents later)
* time of visit
* webpage title
* html (web cache)
* main text of the article (optional)
* structured data required
* semi-structured data required (if a relational database is used, this can be stored as a JSON string)

### code example

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
    # if the webpage is not encoded in UTF-8, encoding of which should be set explicitly
    # r.encoding = 'gb2312'
    html = r.text
    soup = BeautifulSoup(html, 'lxml')
except requests.exceptions.Timeout as e:
    # timeout, put it back in queue and visit it later
except requests.exceptions.RequestException as e:
    # in most cases this url should be aborted
```

```python
links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
title = soup.find(class_='core_title_txt').text
posts = [p.text for p in soup.find_all(id=re.compile("post_content_"))] # id="post_content_137"
next_page_url =
    [urljoin(url, a['href']) for a in soup.find_all('a', href=True) if a.text=='Next Page'][0]
# if the selector in Chrome developer tools is "body > div.mainWarp > div > div.main > div > div > ul.zsList"
soup.body.find('div', class_='mainWarp').div.find('div', class_='main').div.div.find('ul', class_='zsList')
```

## Other tools

- [Python Documentation](https://docs.python.org/3/)
- [Python Standard Library](https://pymotw.com/3/)
- Date & Time: [Arrow](http://arrow.readthedocs.io/en/latest/)
- Language
  - Chinese tokenization：
    - [PKUSeg](https://github.com/lancopku/PKUSeg-python)
    - [THULAC](https://github.com/thunlp/THULAC-Python)
    - [Jieba](https://github.com/LiveMirror/jieba)
  - [Pinyin](https://github.com/lxneng/xpinyin)
  - [traditional <=> simplified Chinese](https://github.com/BYVoid/OpenCC)
- Math
  - [Symbolic computation](http://live.sympy.org/)
- Profiler
  - time: [line profiler](https://github.com/rkern/line_profiler)
  - space: [memory profiler](https://pypi.python.org/pypi/memory_profiler)
- pretty printing in terminal
  - [termcolor](https://github.com/hfeeki/termcolor)
  - [prettytable](https://github.com/vishvananda/prettytable)

## Learning resources for Python

If this is the first time you are trying to use Python, there are many good Python tutorials on the Internet to get you started. Mark Pilgrim’s [Dive Into Python](http://www.diveintopython.net/) is one that I personally suggest. If this is the first time you ever try to use a programming language, A [Byte of Python](http://swaroopch.com/notes/Python) is even better. If you already have a stable programming background in other languages and you just want a quick overview of Python, [Learn Python in 10 minutes](http://www.poromenos.org/tutorials/python) is probably your best bet.

