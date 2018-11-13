FROM phusion/baseimage

ARG PYTHON_VERSION=3

WORKDIR /home/AiLab

RUN apt-get update \
 && apt-get install -y curl gcc g++ python python-dev python3 python3-dev sasl2-bin libsasl2-2 libsasl2-dev libsasl2-modules libsm6 libxrender1 libxext-dev libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN alias python=python${PYTHON_VERSION}

RUN curl https://bootstrap.pypa.io/get-pip.py | python

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install pandas thrift sasl thrift_sasl pyhive oss2 opencv-python imageio

COPY . /home/AiLab

RUN pip install /home/AiLab

CMD [ "bash" ]
