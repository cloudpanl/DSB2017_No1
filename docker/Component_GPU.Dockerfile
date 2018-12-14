FROM pytorch/pytorch

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN apt-get update \
 && apt-get install -y wget gcc g++ sasl2-bin libsasl2-2 libsasl2-dev libsasl2-modules libsm6 libxrender1 libxext-dev libglib2.0-0 \
 && apt-get clean

ADD https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb /tmp/nvcc.deb

RUN dpkg -i /tmp/nvcc.deb && rm -rf /tmp/nvcc.deb

RUN apt-get update \
 && apt-get install -y libnccl2 libnccl-dev \
 && apt-get clean

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN pip install suanpan[docker] pydicom scipy h5py scikit-image simpleitk matplotlib nvidia-ml-py3 \
 && rm -rf ~/.cache/pip

WORKDIR /home/DSB3

COPY . /home/DSB3

CMD [ "bash" ]
