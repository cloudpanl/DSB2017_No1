FROM phusion/baseimage

ARG PYTHON_VERSION=3

WORKDIR /home/AiLab

RUN apt-get update \
 && apt-get install -y curl gcc python python-dev python3 python3-dev sasl2-bin libsasl2-2 libsasl2-dev libsasl2-modules \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN alias python=python${PYTHON_VERSION}

RUN curl https://bootstrap.pypa.io/get-pip.py | python

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install grpcio-tools googleapis-common-protos thrift sasl thrift_sasl pyhive[hive]

COPY . /home/AiLab

RUN pip install /home/AiLab

CMD [ "bash" ]
