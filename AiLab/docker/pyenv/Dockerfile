FROM registry.cn-shanghai.aliyuncs.com/shuzhi/conda_base

RUN mkdir -p /home/AiLab

ARG PYTHON_VERSION=3

ENV PYTHON_VENV_NAME py${PYTHON_VERSION}env

WORKDIR /home/AiLab

RUN alias conda=${CONDA_HOME}/bin/conda

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
 && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
 && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
 && conda config --set show_channel_urls yes

RUN conda create --name ${PYTHON_VENV_NAME} --copy --yes python=${PYTHON_VERSION} pip git \
 && conda install -n ${PYTHON_VENV_NAME} --yes cython

RUN alias pip=${CONDA_HOME}/envs/${PYTHON_VENV_NAME}/bin/pip

RUN pip install --upgrade pip

RUN pip install --upgrade git+https://github.com/jpmml/pyspark2pmml.git \
 && pip install --upgrade pyodps jinja2

COPY . /home/AiLab

RUN pip install /home/AiLab

RUN cd ${CONDA_HOME}/envs \
 && zip -r -9 -q /home/${PYTHON_VENV_NAME}.zip ./${PYTHON_VENV_NAME} \
 && rm -rf ./${PYTHON_VENV_NAME}

RUN mkdir -p /dist

CMD mv -f /home/${PYTHON_VENV_NAME}.zip /dist
