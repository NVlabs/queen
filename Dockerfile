FROM nvcr.io/nvidia/pytorch:23.07-py3

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN conda config --set ssl_verify no
RUN conda install -y python==3.12.0 pytorch==2.5.0 torchvision==0.20.0 -c pytorch -c nvidia
RUN conda install plyfile tqdm -c conda-forge
ADD ./submodules ./submodules
RUN pip install setuptools
RUN pip3.10 install submodules/simple-knn
RUN apt-get update && apt-get install libgl1 -y
RUN pip install opencv-python scipy wandb six
WORKDIR /workspace/submodules/diff-gaussian-rasterization
RUN pip3.10 install .
WORKDIR /workspace/submodules/gaussian-rasterization-grad
RUN apt-get install libglm-dev
RUN pip3.10 install .
WORKDIR /workspace
RUN pip3.10 install torchmetrics imutils einops==0.6.0 timm==0.6.12 matplotlib
COPY ./maxxvit.py /root/miniconda3/lib/python3.11/site-packages/timm/models/maxxvit.py