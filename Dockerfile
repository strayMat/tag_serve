FROM ubuntu:16.04

# Install some basic utilities
RUN http_proxy=$http_proxy \
    apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -x $http_proxy -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment

RUN http_proxy=$http_proxy https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy /home/user/miniconda/bin/conda install conda-build
RUN http_proxy=$http_proxy https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy /home/user/miniconda/bin/conda create -y --name py36 python=3.6.5
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# No CUDA-specific steps
ENV NO_CUDA=1
RUN http_proxy=$http_proxy https_proxy=$http_proxy conda install -y -c pytorch \
    pytorch-cpu=0.4.1 \
    torchvision-cpu=0.2.1 \
 && conda clean -ya

# Install HDF5 Python bindings
RUN http_proxy=$http_proxy  https_proxy=$http_proxy conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN http_proxy=$http_proxy https_proxy=$http_proxy pip install h5py-cache==1.0

# Install Torchnet, a high-level framework for PyTorch
RUN http_proxy=$http_proxy https_proxy=$http_proxy pip install torchnet==0.0.4

# Install Requests, a Python library for making HTTP requests
RUN http_proxy=$http_proxy https_proxy=$http_proxy conda install -y requests=2.19.1 \
 && conda clean -ya

# Install Graphviz
RUN http_proxy=$http_proxy https_proxy=$http_proxy conda install -y graphviz=2.38.0 \
 && conda clean -ya
RUN http_proxy=$http_proxy https_proxy=$http_proxy pip install graphviz==0.8.4

USER root
# Install OpenCV3 Python bindings
RUN http_proxy=$http_proxy apt-get update 
RUN http_proxy=$http_proxy apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \ 
 && sudo rm -rf /var/lib/apt/lists/*

USER user

RUN http_proxy=$http_proxy https_proxy=$http_proxy conda install -y -c menpo opencv3=3.1.0 \
 && conda clean -ya
USER root
RUN http_proxy=$http_proxy \
    apt-get update && apt-get install -y \
    gcc
USER user

RUN http_proxy=$http_proxy https_proxy=$http_proxy pip install \
    numpy \
    scikit-learn \
    tqdm \
    Flask==1.0.2 \
    Flask-Cors==3.0.4 \ 
    gevent==1.3.0

RUN http_proxy=$http_proxy https_proxy=$http_proxy conda install -c conda-forge spacy

RUN http_proxy=$http_proxy https_proxy=$http_proxy python -m spacy download fr

COPY . /app/
WORKDIR /app/

EXPOSE 5000
# Set the default command to python3
CMD ["python3", "/app/app.py ", "-l", "fr"]
