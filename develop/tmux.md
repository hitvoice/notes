[Tmux Reference](http://wiki.fast.ai/index.php/Tmux)
[A Quick and Easy Guide to tmux](http://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/)

```sh
# create
tmux new -s SessionName
# create with more history buffer (default is 2000)
tmux set-option history-limit 5000 \; new-window # new pane in a session
tmux set-option -g history-limit 5000 \; new-session
# attach
tmux a -t SessionName 
# list session
tmux ls 
# terminate
tmux kill-session -t SessionName 
# help
tmux list-keys | less 
man tmux
```
Inside a session, press Ctrl+A, then:
```sh
d      # detach
```
Panel actions:
```sh
|      # vertical splilt
-      # horizontal split
x      # kill panel
Up/Down/Left/Right # move between panels
ESC + Up/Down/Left/Right # resize current panel
[      # enter scroll mode, use arrow up/down/mouse wheel to scroll, press `q` to quit
```
Window actions:
```sh
c      # create window
w      # list windows and select
,      # rename window
&      # kill window
```
Others:
```sh
tmux break-pane # move current panel to a new window
```

### Setup Notes
Setup [mem-cpu-load](https://github.com/thewtex/tmux-mem-cpu-load) (manually).
Create ~/.tmux.conf:
```
unbind C-b
set -g prefix C-a

# split panes using | and -
bind | split-window -h
bind - split-window -v
unbind '"'
unbind %

# show memory and CPU load
set -g status-interval 2
set -g status-left "#S #[fg=gray,bg=black]#(tmux-mem-cpu-load --interval 2 -a 0)#[default]"
set -g status-left-length 50

# default window start number: 1
set-option -g base-index 1

# Reload tmux config
bind r source-file ~/.tmux.conf
```
```sh
tmux source-file ~/.tmux.conf
```