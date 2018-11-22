FROM registry.cn-shanghai.aliyuncs.com/shuzhi/docker_base:3

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN pip install torch torchvision pydicom scipy h5py scikit-image imageio simpleitk

RUN rm -rf ~/.cache/pip

WORKDIR /home/DSB3

COPY . /home/DSB3

CMD [ "bash" ]
