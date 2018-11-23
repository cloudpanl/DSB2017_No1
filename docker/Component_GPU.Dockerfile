FROM pytorch/pytorch

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN apt-get update \
 && apt-get install -y gcc g++ sasl2-bin libsasl2-2 libsasl2-dev libsasl2-modules libsm6 libxrender1 libxext-dev libglib2.0-0 \
 && apt-get clean

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN pip install suanpan[docker] pydicom scipy h5py scikit-image simpleitk matplotlib nvidia-ml-py3 \
 && rm -rf ~/.cache/pip

WORKDIR /home/DSB3

COPY . /home/DSB3

CMD [ "bash" ]
