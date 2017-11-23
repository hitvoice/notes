```sh
# mostly used
git add -A
git commit
git push

cd project_repo; git init # setup a new local repo
git clone xxx.git # clone a remote repo
git status
git status -s
git diff # between staged and unstaged files
git difftool
git add xxx.py # put this file into the project
git add -A # add all the files in the directory
git diff --staged # between staged files and last commit
git commit -m "add xxx.py"
git commit # use vim to enter a multi-line commit message; use after pull and fix a conflict
git push

git pull
git log -3 # view the latest 3 commits
git rm xxx # untrack and remove a file
git rm --cached xxx # untrack
git mv ReadMe ReadMe.md # rename a file

# sync with upstream repo when working on a forked one
git remote add upstream https://xxx.git
git remote -v # remote info
git remote set-url origin https://xxx.git # change it later
git fetch upstream
git merge upstream/master
git push

git branch # show all the branches
git branch feature_x # create a new branch copied from master
git checkout feature_x # switch to another branch (or commit)
git checkout master # return to master branch, or from a previous commit (detached head)
git checkout -b feature_x # create a branch and switch to it
git checkout -b old-state 0d1d7fc32 # create a branch based on a previous commit
git merge master # merge changes from master into the current branch
git branch -d feature_x # delete the branch
# 多说几句：如果已经做了一些改动然后希望把这些改动放在新branch里，master回到上一次commit，做法是新建一个branch（此时修改在两个branch都可见），然后在新branch里commit changes，master就会自动回到上一次commit，而branch更新为修改后的状态

git tag -a v1.4 -m "message for this tag" # will launch vim without `-m`
git show v1.4 # view tag info
git push origin --tags

git commit --amend # commit again and overwrite the last commit (when you commit too early)
# DANGEROUS: the following command can not be undone
git checkout -- xx.py # discard recent changes and go back to the latest commit
git checkout -- . # discard all recent changes and go back to the latest commit
git reset --hard master@{"10 minutes ago"} # recover from some terrible mistake
```
work with git-lfs:
```sh
# install (on Mac)
brew install git-lfs
# (on Ubuntu)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

# add track files
git lfs track "*.psd"
# (optional) link to SourceTree
ln -s /usr/local/bin/git-lfs /Applications/SourceTree.app/Contents/Resources/git_local/bin
```
set vim as the default editor (sometimes the default editor is nano):
```sh
git config --global core.editor "vim"
```
set Xcode FileMerge (opendiff) as default difftool:
```sh
sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer
git config --global merge.tool opendiff
```

### .gitignore
.gitignore format：

* Blank lines or lines starting with \# are ignored
* Standard glob patterns work
  * An asterisk (\*) matches zero or more characters
  * [abc] matches any character inside the brackets (in this case a, b, or c)
  * a question mark (?) matches a single character
  * brackets enclosing characters separated by a hyphen (e.g. [0-9]) matches any character between them
  * use two asterisks to match nested directories; a/\*\*/z would match a/z, a/b/z, a/b/c/z, and so on.
* start patterns with a forward slash (/) to avoid recursivity
* end patterns with a forward slash (/) to specify a directory
* negate a pattern by starting it with an exclamation point (!)

example .gitignore file:
```
# no .a or .o files
*.[oa]

# but do track lib.a, even though you're ignoring .a files above
!lib.a

# only ignore the TODO file in the current directory, not subdir/TODO
/TODO

# ignore all files in the build/ directory
build/

# ignore doc/notes.txt, but not doc/server/arch.txt
doc/*.txt

# ignore all .pdf files in the doc/ directory and any of its subdirectories
doc/**/*.pdf
```
```sh
# account settings
git config user.name # show username
git config user.email

git config --global user.email "xxx@gmail.com"
git config --local user.email "xxx@gmail.com" # repository
```