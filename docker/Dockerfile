FROM registry.cn-shanghai.aliyuncs.com/shuzhi/docker_component:1.6

RUN pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl \
 && pip install torchvision pydicom scipy h5py scikit-image imageio simpleitk matplotlib==2.0.0

WORKDIR /home/DSB3

COPY . /home/DSB3

CMD [ "bash" ]
