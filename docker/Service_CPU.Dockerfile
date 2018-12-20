FROM registry.cn-shanghai.aliyuncs.com/shuzhi/service_base:3

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN pip install --no-cache-dir torch torchvision pydicom scipy h5py scikit-image imageio simpleitk

WORKDIR /home/DSB3

RUN mkdir -p /home/DSB3

COPY dsb /home/DSB3/dsb

COPY service_* /home/DSB3/

CMD [ "bash" ]
