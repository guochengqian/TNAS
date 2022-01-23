#!/usr/bin/env bash
# make sure command is : source env_install.sh

# install anaconda3.
# cd ~/
# wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
# bash Anaconda3-2019.07-Linux-x86_64.sh

# module load, uncommet if using local machine
#module purge
#module load gcc
#module load cuda/10.1.105

# make sure your annaconda3 is added to bashrc
#source activate
#source ~/.bashrc

conda create --name tnas 
conda activate tnas
conda install -y pytorch=1.7.0 torchvision cudatoolkit=10.1 python=3.6.8 Pillow=6.1 -c pytorch 

# install useful modules
pip install tqdm graphviz tensorboard wandb easydict multimethod nats-bench gdown

python setup.py install  # install xautodl 


# download the NATS-Bench file
export TORCH_HOME=~/.torch
mkdir ~/.torch
cd $TORCH_HOME
gdown https://drive.google.com/uc?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU
gdown https://drive.google.com/uc?id=1scOMTUwcQhAMa_IMedp9lTzwmgqHLGgA

tar -xf NATS-tss-v1_0-3ffb9-simple.tar
tar -xf NATS-sss-v1_0-50262-simple.tar
