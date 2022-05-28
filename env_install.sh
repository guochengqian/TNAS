#!/usr/bin/env bash
# install miniconda3 if not installed yet. 
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
#bash Miniconda3-latest-Linux-x86_64.sh  
#source ~/.bashrc

# module load, only necessary for slurm
module purge
module load gcc
module load cuda/11.1.1

export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"   # a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1

conda deactivate
conda env remove --name tnas
conda create -n tnas python=3.7 numpy=1.20 numba -y 
conda activate tnas
conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.1 python=3.6.8 Pillow=6.1 -c pytorch 

# install useful modules
pip install tqdm graphviz tensorboard wandb easydict multimethod nats-bench gdown termcolor
python setup.py install  # install xautodl 

# download the NATS-Bench file
export TORCH_HOME=~/.torch
mkdir ~/.torch
cd $TORCH_HOME
gdown https://drive.google.com/uc?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU
gdown https://drive.google.com/uc?id=1scOMTUwcQhAMa_IMedp9lTzwmgqHLGgA
tar -xf NATS-tss-v1_0-3ffb9-simple.tar
tar -xf NATS-sss-v1_0-50262-simple.tar