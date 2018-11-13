FROM phusion/baseimage

RUN apt-get update \
 && apt-get install -y curl zip unzip vim sasl2-bin libsasl2-2 libsasl2-dev libsasl2-modules build-essential git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN curl 'https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh' -o /tmp/c.sh

RUN sh /tmp/c.sh -b -p /home/anaconda && rm /tmp/c.sh

ENV CONDA_HOME /home/anaconda

CMD sleep infinity
