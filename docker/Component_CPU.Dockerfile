FROM registry.cn-shanghai.aliyuncs.com/shuzhi/docker_component:3

ARG PYPI_MIRROR="https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-linux_x86_64.whl \
 && pip install torchvision pydicom scipy h5py scikit-image imageio simpleitk

WORKDIR /home/DSB3

COPY . /home/DSB3

CMD [ "bash" ]
