## system
### system and device info
```sh
df -h # disk info 
free -h # memory info
lscpu # CPU info
htop # activity monitor
nvidia-smi # GPU
watch -n 1 -d nvidia-smi # refresh every second and highlight the changes
nvcc --version # CUDA version
lsb_release -a # ubuntu system version
ifconfig # IP address
lsblk # disk space
lsblk --fs # file system and label of the disks
sudo parted -l # detailed info of disks
sudo netstat -tulpn # active ports and which are listening
lsof -i # list the process (pid) running on port
ps aux | grep java # list all processes whose commands contain "java"
```
### external drive
```sh
# mount a USB drive
lsblk # Find the path to the drive
sudo mkdir /media/usb # Create a mount point
sudo mount /dev/sdb1 /media/usb # Mount!
sudo umount /media/usb # fire off after using

# format and mount an external disk
lsblk # Find the path to the disk
sudo parted /dev/sdb mklabel gpt # create partion table and use GPT standard
sudo parted -a opt /dev/sdb mkpart primary ext4 0% 100% # create a new partition
# use file system ext4: best if the disk is linux only
sudo mkfs.ext4 -L "some label" /dev/sdb1 # don't forget the tailing number. We're referring to a partition now
# use file system exfat: ready for Windows, MacOS to read data from it in the future
sudo apt install exfat-utils exfat-fuse
sudo mkfs.exfat -n "some label" /dev/sdb1
lsblk --fs # check the fstype, UUID and label
# mount the partition on boot
sudo mkdir /mnt/data
sudo vi /etc/fstab # add this line: "UUID=B866-DD3A /mnt/data/ ext4 defaults 0 2"
sudo mount -a # mount it now
sudo chown -R username:username /mnt/data/ # gain permission
```
## file
### file transfer
```sh
# download
wget -c http://URL.HERE -O output/file # -c means continue downloading if it's interrupted before
# advanced usage of wget: https://www.thegeekstuff.com/2009/09/the-ultimate-wget-download-guide-with-15-awesome-examples

# speed up scp in a secure environment
# in target machine:
echo "Ciphers $(ssh -Q cipher localhost | paste -d , -s -)" | sudo tee --append /etc/ssh/sshd_config
sudo service sshd restart ; sudo service sshd status
# in source machine
scp -rp -C -o 'CompressionLevel 9' -o 'IPQoS throughput' -c arcfour srcDir tgtDir

# cp/mv big files with progress bar
# build "progress" following https://github.com/Xfennec/progress
cp bigfile newfile & progress -mp $!
```
### zip and unzip
```sh
zip -rq targetFile.zip srcDir # zip quietly
unzip -q xxx.zip # unzip quietly
tar -zcvf xxx.tar.gz srcDir # remove "v" to tar quietly
tar -zxvf xxx.tar.gz # remove "v" to untar quietly
tar -zxvf xxx.tar.gz -C /output/dir # untar to target directory 
gzip xxx # -> xxx.gz, cannot be applied to a directory
gzip -dk xxx.gz  # unzip a gz file. use "-k" to keep the original file

# zip with a progress bar
SRCDIR=dir_with_big_files
tar cf - $SRCDIR -P | pv -s $(du -sb $SRCDIR | awk '{print $1}') | gzip > $SRCDIR.tar.gz
zip -qr - $SRCDIR | pv -s $(du -bs $SRCDIR | awk '{print $1}') > $SRCDIR.zip

# unzip with a progress bar
pv archive.tar.gz | tar -xz 
pv archive.tar | tar -x
unzip archive.zip | pv -l >/dev/null

# zip into pieces and unzip
tar czpvf - /path/to/archive | split -d -b 100M - fileprefix
cat fileprefix* | tar xzpvf -
```
### file info
```sh
head -10 file.txt # print the first 10 lines of a file
tail file.txt 
tail -f file.txt # wait for another program to write and print new contents
cat file.txt # print entire file
less file.txt # open a previou "window" for the file
wc -l # number of lines in a file
wc -L # maximum line length in a file
# average length of a file
awk ' { thislen=length($0); totlen+=thislen} END { printf("average: %d\n", totlen/NR); } ' file.txt
feh image.jpg # open an image from the terminal (or multiple images at the same time)
```
### common file processing
```sh
# file creation
touch file.txt # create if not exists; otherwise update timestamp
echo -n > file.txt # create if not exists; otherwise truncate to empty
echo -n >> file.txt # same as `touch file.txt`

# create a file shortcut/symbolic file (use absolute path to avoid potential problems)
ln -s src des 

# filter lines of a file
grep -i "warn" main.log > warn.log # -i case insensitive
grep -E "[0-9.]+$" main.log > num.log  # -E uses regular expression (or "egrep")
grep -oE "[0-9.]+$" main.log > num.log  # -o only the matched part instead of the whole line
grep -E $'^[^\t]+\t[^\t]+\t\d$' data.txt > data.clean.txt  # use $'' to enclose special chars
grep -vE $'^[^\t]+\t[^\t]+\t\d$' > bad_lines.txt # select lines NOT matching the regex

# shuffle the lines of a file
myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}
cat file.txt | myshuf > output.txt

# remove duplicatd lines in a file
awk '!a[$0]++' input.txt > input_unique.txt
sort -u input.txt > input_unique.txt  # slower, only when if you need to sort as well

# select duplicated lines in a file with counts
sort input.txt | uniq -cd | sort -nr

# split a file into equally-sized chunks
split -l 1000 -d -a2 input.txt prefix_  # each file will be named prefix_01, prefix_02 containing 1000 lines 
split -C 20m -d -a2 input.txt prefix_  # at most 20MB (not breaking lines)
cat input_* | split -l 1000 -d -a2 - prefix_  # merge and split again

# merge multiple files
cat prefix_*.txt > full.txt
cat file1 file2 > full.txt

# train-test split by order
head -n -2000 $file > train.txt
tail -n 2000 $file > dev.txt
# random train-test split 
shuf $file | split -a1 -d -l $(( $(wc -l <$file) - 2000 )) - output
mv output0 train.txt
mv output1 dev.txt

# extract random samples from a file
shuf -n 2000 input.txt > sample.txt

# extract line n~n+m-1 from a file
tail +n $file | head -m 

# extract the different parts between 2 files
diff file1.txt file2.txt | grep "^>" | cut -c 2- > diff1.txt
diff file1.txt file2.txt | grep "^<" | cut -c 2- > diff2.txt

# cleanup newlines when using files from Windows (\r, displayed as extra "^M" in vim)
sed -i -e 's/\r//g' input.txt # inplace

# remove UTF8 BOM (<U+FEFF>)
sed -i '1s/^\xEF\xBB\xBF//' orig.txt # inplace
sed '1s/^\xEF\xBB\xBF//' < orig.txt > new.txt # new file
```

## directory
### directory info
```sh
du -hs * # disk space taken up by each file and directory
ls -lGFh # disk space taken up by every directory, sub-directory and file
ls -U | head -4 # print the first 4 files of a folder (with numerous amount of files)
ls -lhS # sorted by file size
find . -type f -printf . | wc -c # count the number of files (including hidden files) in a directory
tree -d # print the directory stucture as a tree
```

### search and batch processing of files
```sh
# search files
find srcDir -name *.jpg

# move a large amount of files (which exceeds the maximum number permitted by *)
find srcDir -name '*.jpg' -exec mv {} targetDir \;
#                                  {} will be filled by each result of "find"
#                                               \ denotes the ending of "-exec" commands
# find files in srcDir recursively. refer to the manual of "find" for how to limit the maximum recursion depth

# data sampling
mkdir samples
myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}
find train/ -name '*.jpg' | myshuf | head -20 | xargs cp -t samples/

# batch renaming with regex expressions
# example 1: add ".gif" to all files in the current directory
rename 's/(.+)/$1.gif/' *
# exmaple 2: rename "[Student_ID] [Name].jpg" to "[Student_ID].jpg"
rename 's/(\d{9}).+(\.jpg)/$1$2/' *.jpg
# on some systems rename doesn't support regex, for example 1 you can simply do
for x in *;do mv $x $x.gif; done

# remove all the empty files in the current directory
find . -size 0 -print0 | xargs -0 rm
```
## variables and control flow
```sh
# variables
x=$(date) # execute the command and assign the output to the variable, equals to x=`date`
${x} # use the value of the variable

# std IO
printf "%s\n" "$x"
read -p "Enter your name : " name # add `-s` for passwords
read -p "enter: " -r v1 v2 v3 # use space to separate multiple inputs
read -r x1 x2 <<< "3 1" # redirect

# for loop
for i in `seq 1 30`; do
    echo $i
    python process.py ${i}.txt
done

# loop over tuples
for i in in1,out1 in2,out2 ; do 
    IFS=',' read a b <<< "${i}"
    echo $a-$b  # in1-out1
done

# while loop (read from stdin)
ls | grep -E *.txt | while read -r line ; do upload $line ; done

# check script input argument
if [[ $# -eq 0 ]] ; then
    echo 'error: no argument provided'
    exit 1
fi

# regex example
files="models/ssd_mobilenet_train/*.meta"
regex="ckpt\-([0-9]+)\.meta"
for f in $files ; do
    if [[ $f =~ $regex ]] ; then
        name="${BASH_REMATCH[1]}" # get the model id (group 1)
        mkdir -p dumps/$name
        # command to dump the model (omitted)
    fi
done
```
## other utilities
```sh
alias ssh-s1='ssh -i ~/ubuntu.pem ubuntu@xx.xx.xx.xx'
type ssh-s1 # print what's bound to the alias

# write output of a program to both stdout and file 
command | tee output.log  # only stdout
command | tee -a output.log  # only stdout (append to output.log)
command |& tee output.log  # stdout + stderr
command > >(tee stdout.log) 2> >(tee stderr.log >&2)  # write to separate files

```

## vim basics
- count number of matches `:%s/pattern//ng`
- print all lines that match a pattern `:g/pattern/p`
- paste without indent `:set paste`; reset `:set nopaste`

## resources
- [tutorial](https://bash.cyberciti.biz/guide/Main_Page)
