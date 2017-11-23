# Docker

（安装指令见底部）

常用命令
----

### info

`docker images`

`docker ps -sl`

`docker port CONTAINER_NAME`

`docker top CONTAINER_NAME` (processes running inside)

`docker logs CONTAINER_NAME`

`docker inspect CONTAINER_NAME`

`docker volume ls`

[visualize image](https://imagelayers.io)

### run

(interactive)

new container:

`docker run -it —-rm --name [container-name] [image-name]`​ (leave the console using ​`exit`​)

a concrete example with a shared folder:

`docker run -it —-rm -v `pwd`:/home/data utkarshp/kenlm`

resume:

`​docker start `[container-name]

docker attach [container-name]

run a script inside a container:

`​docker run -it —-rm -v `pwd`:/home/data `[image-name] /bin/sh -c “YOUR COMMANDS”`​`​ (`​`​start at location “/“​)

(daemon)

`docker run -d —-rm ubuntu /bin/sh -c "while true; do echo hello world; sleep 1; done”`

(web)

`docker run -d -p 80:5000 training/webapp python app.py` (container:5000-\>localhost:80)

### action

`docker stop CONTAINER_NAME`

`docker start CONTAINER_NAME`

`docker restart CONTAINER_NAME`

`ddocker rm CONTAINER_NAME`

(DANGER: clean up all) `docker rm docker ps --no-trunc -aq` 

### build

`docker build -t repo/name:ver .`

`docker commit -m "msg" -a "author" container-id repo/name:v2`

### remove

`docker rmi -f image-name`

### publish

`docker tag image-id maryatdocker/docker-whale:latest`

`docker login`

`docker push maryatdocker/docker-whale` 

Dockerfile
----------

```
# This is a comment
FROM ubuntu:14.04
MAINTAINER Kate Smith <ksmith@example.com>
RUN apt-get update && apt-get install -y ruby ruby-dev
RUN gem install sinatra

```

    启动：(Mac) 运行docker桌面版程序 （linux）[安装指南](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04)

    [国内镜像加速](https://www.docker-cn.com/registry-mirror)

### 安装nvidia-docker

```sh
wget https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i nvidia-docker*.deb && rm nvidia-docker*.deb
nvidia-docker run --rm nvidia/cuda nvidia-smi
```

### 不用sudo执行docker

```sh
sudo usermod -aG docker ${USER}
```

然后在命令行logout再login，如果是桌面系统那么需要重启。可以用如下命令检查是否加入了docker group：

```sh
id -nG
```