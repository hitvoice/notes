## Basics
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
git reset HEAD xxx  # change a file from staged to unstaged
git reset  # unstage all files
git reset --soft HEAD^  # cancel the latest commit (but keep the changes)
git rm xxx # untrack and remove a file
git rm --cached xxx # untrack
git mv ReadMe ReadMe.md # rename a file

# add remote origin
git remote add origin https://xxx.git
git remote -v # remote info
git remote set-url origin https://xxx.git # change it later
git remote set-url --push origin https://xxx.git
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
git merge --abort # if you encounter a merge conflict by mistake or you do not want to handle it now
git branch -d <branch_name> # delete a local branch
git push -d origin <branch_name> # delete a remote branch

git submodule add <URL> <path>
git submodule update # when git status says: "modified: xxx (new commits)"
git tag # list your tags
git tag -l v1.8.* # list tags with prefixes
git tag -a v1.4 -m "message for this tag" # will launch vim without `-m`
git tag v1.4 # create a lightweight tag without annotations
git tag -a v1.2 9fceb02 # tag previous commit 9fceb02
git tag -d v1.4-lw  # delete a tag
git show v1.4 # view tag info
git push origin <tag_name> # push a single tag

git commit --amend # commit again and overwrite the last commit (when you commit too early)
# revert a range of commits (from old to new). Unstaged files not affected by these commits will remain untouched
git revert --no-commit a867b4af..0766c053 
# DANGEROUS: the following command can not be undone
git checkout -- xx.py # discard recent changes and go back to the latest commit
git checkout -- . # discard all recent changes and go back to the latest commit
git chekcout <commit-sha> . # return to a certain commit. New files will remain untouched but unstaged modificatoins will be lost
git reset --hard master@{"10 minutes ago"} # recover from some terrible mistake
git reset --hard <commit-sha> # go back to this commit and discard all following changes
```
If some changes have been made to files in the master branch, but you somehow decide to keep the master branch untouched and put the changes in a new branch, you should create a new branch (now the changes are visible in both branches) and commit in the new branch. The master branch will fall back to its last commit.

If some changes have been made to files in the master branch, but you want to switch to another branch and apply changes in that branch, or to pull remote changes first, you can use `git stash`. After switching to another branch or pulling remote changes, use `git stash pop` to apply previous modifications. `git stash drop` can be used to drop stash that's no longer needed. Use `git stash show` to show the files that changed in your most recent stash. Use `git stash show -p` to show the diff.

If in some scenarios, master branch is deprecated and some other branch should be the new master, do the following:
```sh
# https://stackoverflow.com/questions/2763006/make-the-current-git-branch-a-master-branch
git checkout better_branch
git merge --strategy=ours master -m "massage'   # keep this branch untouched, but record a merge
git checkout master
git merge better_branch                         # fast-forward master up to the merge
```
## commit message conventions
Here's a [example convention](https://gist.github.com/stephenparish/9941e89d80e2bc58a153) from AngularJS.

## work with git-lfs:
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
## editor and difftool
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

## .gitignore
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
## account settings
### in commit messages
```sh
git config user.name # show username
git config user.email

git config --global user.email "xxx@gmail.com"
git config --local user.email "xxx@gmail.com" # repository
```
To change the author after a commit, first set the correct account in the current repo, then use `git commit --amend --reset-author --no-edit`. Use `git log` to check the result.

### authentication - password
If you're on a completely private server, `git config credential.helper store` can be used to store the password for the remote server. NEVER use this command on a shared server because the password is stored in plain text.

When using Github, go to "Settings->Developer settings->Personal access tokens" to generate to new token to use instead of the explicit password.

To update the password/acees token, use `git config --global credential.helper osxkeychain` on MacOS and it will prompt the next time you need authenication. 

### authentication - SSH
To compare fingerprints of SSH key in the local machine with the Github remote one, use `ssh-keygen -lf ~/.ssh/id_rsa -E sha256`.

[Setup multiple Github accounts on a single machine](https://gist.github.com/JoaquimLey/e6049a12c8fd2923611802384cd2fb4a)

If the authentication failure relates to linking to a wrong account:
- check ~/.ssh/config to see if the correct hostname is used
- use `ssh -T git@github.com` to verify the current user, `ssh -vT git@github.com` to see debug logs
- if the wrong account comes from the ssh agent, use `ssh-add -D` to clear its cache

## working on Windows
To supress warnings about different line separators:
```sh
git config core.eol lf
git config core.autocrlf
```
## Trouble shooting
"The unauthenticated git protocol on port 9418 is no longer supported."
It can happen when using something like pre-commit. Use `git config --global url."https://".insteadOf git://` to change the default protocol.
