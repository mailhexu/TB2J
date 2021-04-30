#!/usr/bin/env sh
if ! [ -x "$(command -v docker)" ]; then
	cat << ENDOFOUTPUT
You need to install docker ( https://www.docker.com/ ) to build a docker image'

On a Ubuntu 20.04 the commands below are the best way to do it.
There are methods that are a lot easier, but this is the best:

#As root run (replace YOURUSERNAME by your username):

apt-get -y install apt-transport-https ca-certificates curl software-properties-common && \\
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - && apt-key fingerprint 0EBFCD88 && \\
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable" && \\
apt-get update && apt-get -y install docker-ce && usermod -G docker -a YOURUSERNAME && \\
echo '{ "storage-driver": "overlay2" }' > /etc/docker/daemon.json && \\
apt-get -y install docker-compose docker-ce-cli containerd.io
ENDOFOUTPUT
	exit 1
fi
if [ ! -z $1 ] &&  [ $1 = "build" ] ; then
	cat << ENDOFDOCKERFILE  > Dockerfile
FROM python:3
RUN pip3 install TB2J
WORKDIR /root
ENDOFDOCKERFILE
	if [ ! -z $2 ] &&  [ $2 = "withrepo" ] ; then
		cat << ENDOFDOCKERFILE  >> Dockerfile
RUN git clone https://github.com/mailhexu/TB2J
ENDOFDOCKERFILE
	fi
	$(command -v docker) build -t tb2j .
	rm Dockerfile
elif [ ! -z $1 ] &&  [ $1 = "run" ] ; then
	$(command -v docker) run -it --name tb2j tb2j /bin/bash
elif [ ! -z $1 ] &&  [ $1 = "rm" ] ; then
	$(command -v docker) rm tb2j
elif [ ! -z $1 ] &&  [ $1 = "rmi" ] ; then
	$(command -v docker) rmi tb2j
else
	cat << ENDOFOUTPUT
$0 build          -> Build the 'tb2j' dockerimage
$0 build withrepo -> Also include the git repo in the image (contains examples)
$0 run            -> Run bash in a new container based on this image
$0 test           -> Runs 'TB2J/examples/abinit-w90/SrMnO3/get_J.sh' as test.
                     (This only works when the image is build with 'withrepo')
$0 rm             -> Remove the container after stopping it
$0 rmi            -> Remove the image

I suggest to only build with this script and do the running/removing manually
ENDOFOUTPUT
fi
