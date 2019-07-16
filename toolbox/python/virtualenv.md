使用conda环境
---------

```sh
conda info --envs // or `conda env list`
conda create -n env_name python=3.5
activate env_name
deactivate
conda remove --name env_name --all // or `conda env remove --name env_name`
```
注意此时ipython仍然是全局的ipython，如果要在虚拟环境内使用ipython，在虚拟环境内先“conda install ipython”。

使用virtualenv [文档](http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/)
--------------------------------------------------------------------------------------

### 基于每个project的本地管理

安装：
```sh
conda install virtualenv
```
创建：

```sh
virtualenv envname
# virtualenv -p /usr/bin/python2.7 envname
# virtualenv --no-site-packages envname
```

此时会把当前环境（指定的环境，或空白环境）copy过来，在当前文件夹下出现”envname”文件夹。接下来使用

```sh
source envname/bin/activate
```

来激活虚拟环境。接下来可以用正常手段用pip安装软件，如：

```sh
pip install keras==1.2
```

需要结束当前环境：

```sh
deactivate
```

如果报错“virtualenv is not compatible with this system or executable”：

```sh
sudo pip uninstall virtualenv
conda install virtualenv
```

### 基于全局的环境管理

安装：

```sh
pip install virtualenvwrapper
export WORKON_HOME=~/Envs # 用来存放虚拟环境
source /Users/runqi/anaconda/bin/virtualenvwrapper.sh
# 非anaconda的Ubuntu存放在/usr/local/bin/virtualenvwrapper.sh
```

基本使用：

```sh
mkvirtualenv envname # 创建
workon envname # 激活
deactivate # 结束
rmvirtualenv envname # 移除
```

### 在Jupyter Notebook中使用不同的虚拟环境 [文档](http://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments)

激活虚拟环境后，使用：

```sh
python -m ipykernel install --user --name <envname> --display-name "envname"
```

即可在kernel列表中选用。如果要移除已经安装的kernel，在`/Users/runqi/Library/Jupyter/kernel`s下把以环境名命名的文件夹删除即可。其他平台上的存放地址可以参考[这里](http://jupyter-client.readthedocs.io/en/latest/kernels.html#kernelspecs)。

