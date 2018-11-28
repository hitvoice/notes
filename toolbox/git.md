### Basics
```sh
# mostly used
git add -A
git commit
git push

cd project_repo; git init # setup a new local repo
git clone xxx.git # clone a remote repo
git clone xxx.git new_name # clone a remote repo using "new_name" as the folder name
git clone -b <branch_name> xxx.git  # clone a specific branch
git submodule update --init # if this repo has submodules, do it after cloning or use "git clone --recursive"
git status
git status -s
git diff # between staged and unstaged files
git difftool
git add xxx # put this file into the project
git add -A # add all the files in the directory
git diff --staged # between staged files and last commit
git commit -m "add xxx"
git commit # use vim to enter a multi-line commit message; use after pull and fix a conflict
git push

git log -3 # view the latest 3 commits
git reset xxx  # change a file from staged to unstaged
git reset  # unstage all files
git rm xxx # untrack and remove a file
git rm --cached xxx # untrack
git mv ReadMe ReadMe.md # rename a file

# add remote origin
git remote add origin https://xxx.git
git remote -v # remote info
git remote set-url origin https://xxx.git # change it later
git pull # update from origin
git submodule update --remote --recursive

# sync with upstream repo when working on a forked one
git remote add upstream https://xxx.git
git remote -v # remote info
git remote set-url upstream https://xxx.git # change it later
git fetch upstream
git merge upstream/master
git push

git branch # show all local branches
git branch -r # show all remote branches
git branch <branch_name> # create a new branch copied from master
git checkout <branch_name/commit_sha> # switch to another branch (or commit)
git checkout master # return to master branch, or from a previous commit
git checkout -b <branch_name> # create a branch and switch to it
git checkout -b <branch_name> <branch_name/commit-sha> # create a branch based on another branch or a previous commit
git merge master # merge changes from master into the current branch
git branch -d <branch_name> # delete the branch

git submodule add <URL> <path>
git tag -a v1.4 -m "message for this tag" # will launch vim without `-m`
git show v1.4 # view tag info
git push origin --tags

git commit --amend # commit again and overwrite the last commit (when you commit too early)
# revert a range of commits (from old to new). Unstaged files not affected by these commits will remain untouched
git revert --no-commit a867b4af..0766c053 
# DANGEROUS: the following command can not be undone
git checkout -- xx.py # discard recent changes and go back to the latest commit
git checkout -- . # discard all recent changes and go back to the latest commit
git chekcout <commit-sha> . # return to a certain commit. New files will remain untouched but unstaged modificatoins will be lost
git reset --hard master@{"10 minutes ago"} # recover from some terrible mistake
```
If some changes have been made to files in the master branch, but you somehow decide to keep the master branch untouched and put the changes in a new branch, you should create a new branch (now the changes are visible in both branches) and commit in the new branch. The master branch will fall back to its last commit.

If some changes have been made to files in the master branch, but you want to switch to another branch and apply changes in that branch, or to pull remote changes first, you can use `git stash`. After switching to another branch or pulling remote changes, use `git stash pop` to apply previous modifications.

If in some scenarios, master branch is deprecated and some other branch should be the new master, do the following:
```sh
# https://stackoverflow.com/questions/2763006/make-the-current-git-branch-a-master-branch
git checkout better_branch
git merge --strategy=ours master -m "massage'   # keep this branch untouched, but record a merge
git checkout master
git merge better_branch                         # fast-forward master up to the merge
```
### work with git-lfs:
```sh
# install (on Mac)
brew install git-lfs
# (on Ubuntu)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

# add track files
git lfs track "*.psd"

# list track files
git lfs ls-files

# (optional) link to SourceTree
ln -s /usr/local/bin/git-lfs /Applications/SourceTree.app/Contents/Resources/git_local/bin
```
### editor and difftool
set vim as the default editor (sometimes the default editor is nano):
```sh
git config --global core.editor "vim"
```
set Xcode FileMerge (opendiff) as default difftool:
```sh
sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer
git config --global merge.tool opendiff
```
set PyCharm as default difftool: 

In PyCharm, select "Tools | Create Command-line Launcher", use default settings. Make sure that the command line tool is in PATH. Add the following line to "~/.gitconfig":
```
[diff]
        tool = pycharm
[difftool "pycharm"]
        cmd = /usr/local/bin/charm diff "$LOCAL" "$REMOTE" && echo "Press enter to continue..." && read
[merge]
        tool = pycharm
[mergetool "pycharm"]
        cmd = /usr/local/bin/charm merge "$LOCAL" "$REMOTE" "$BASE" "$MERGED"
        keepBackup = false
```

### commit message conventions
Here's a [example convention](https://gist.github.com/stephenparish/9941e89d80e2bc58a153) from AngularJS.

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
### account settings
```sh
git config user.name # show username
git config user.email

git config --global user.email "xxx@gmail.com"
git config --local user.email "xxx@gmail.com" # repository
```
### working on Windows
To supress warnings about different line separators:
```sh
git config core.eol lf
git config core.autocrlf
```
