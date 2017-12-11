```sh
df -h #查看磁盘使用情况
free -h #查看内存使用情况
lscpu #查看CPU情况
top # 性能监视器，建议用htop更可读
nvidia-smi # GPU
nvidia-smi -l 1 # 每秒刷新一次
watch -n 1 -d nvidia-smi # 每秒刷新并高亮变化的部分
nvcc --version # CUDA版本
lsb_release -a # ubuntu系统版本
ifconfig # 查看当前IP地址
```

```sh
sudo apt-get install unzip
zip -rq targetFile.zip srcDir # 安静地压缩文件夹
unzip -q xxx.zip # 安静地解压
tar -zcvf xxx.tar.gz srcDir # 去掉v可以安静地压缩/解压
tar -zxvf xxx.tar.gz # -C /output/dir 可以解压到预先创建好的文件夹

# 程序同时输出到屏幕和文件
command | tee output.log

# mount a USB drive
lsblk # Find what the drive is called
sudo mkdir /media/usb # Create a mount point
sudo mount /dev/sdb1 /media/usb # Mount!
sudo umount /media/usb # fire off after using
```

```sh

# 看文件夹
du -hs * # 查看文件夹占用空间
ls -lGFh # 查看文件夹下各一级子文件夹和文件占用空间
ls -U | head -4 #查看一个文件巨多的文件夹
tree -d # 查看目录结构(需要额外安装)

# 看文件
head
tail # 用-f不马上回到命令行，而是等待别的程序写入新内容并显示
cat # 遇见大文件要小心
less # 打开一个新窗口滚动查看文件
wc -l # 文件行数
wc -L # 文件最长行的长度

# 查找/搜索文件
find srcFolder -name *.jpg
# 移动大量文件 (直接用*超出了bash的最大长度)
find srcFolder -name '*.jpg' -exec mv {} targetFolder \;
#                                     {} find结果填入的位置
#                                                     \标志-exec命令的结束位置
# 会递归查找srcFolder下面所有符合要求的文件，如果要限制最大深度，参考find文档
# 数据采样
mkdir samples
find train/ -name '*.jpg' | head -20 | xargs cp -t samples
ln -s src des # 创建快捷方式，用相对路径容易出问题，最好都用绝对路径


# 用正则表达式批量重命名
# 例子：给当前文件夹下所有文件加上.gif后缀
rename 's/(.+)/$1.gif/' *
# 例子：将该文件夹下'学号 姓名.jpg'重命名为'学号.jpg'
rename 's/(\d{9}).+(\.jpg)/$1$2/' *.jpg

# 去除当前目录下所有空文件
find . -size 0 -print0 | xargs -0 rm

alias ssh-qc='ssh -i ~/ubuntu.pem ubuntu@xx.xx.xx.xx'
type ssh-qc # 查看绑定了什么命令
```

```sh
# variables
x=$(date) # 执行指令并返回到变量，等同于 x=`date`
${x} # 引用变量

# std IO
printf "%s\n" "$x"
read -p "Enter your name : " name # add `-s` for passwords
read -p "enter: " -r v1 v2 v3 # use space to separate multiple inputs
read -r x1 x2 <<< "3 1" # redirect

# append a new line
echo 'a new line' >> sample.txt

# for loop
for i in `seq 1 30`;
do
    echo $i
    python process.py ${i}.txt
done

# check script input argument
if [[ $# -eq 0 ]] ; then
    echo 'error: no argument provided'
    exit 1
fi

# regex example
files="models/ssd_mobilenet_train/*.meta"
regex="ckpt\-([0-9]+)\.meta"
for f in $files
do
    if [[ $f =~ $regex ]]
    then
        name="${BASH_REMATCH[1]}" # get the model id (group 1)
        mkdir -p dumps/$name
        # command to dump the model (omitted)
    fi
done
```

## resources
- [tutorial](https://bash.cyberciti.biz/guide/Main_Page)
