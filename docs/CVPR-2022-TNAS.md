
# [TNAS]()

<img align="right" src="http://xuanyidong.com/resources/paper-icon/CVPR-2019-GDAS.png" width="300">




## Requirements and Preparation

```bash
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

```




## Usage

### Reproducing the results of our searched architecture in GDAS
Please use the following scripts to train the searched GDAS-searched CNN on CIFAR-10, CIFAR-100, and ImageNet.
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar10  GDAS_V1 96 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts/nas-infer-train.sh cifar100 GDAS_V1 96 -1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/nas-infer-train.sh imagenet-1k GDAS_V1 256 -1
```
If you are interested in the configs of each NAS-searched architecture, they are defined at [genotypes.py](https://github.com/D-X-Y/AutoDL-Projects/blob/main/xautodl/nas_infer_model/DXYs/genotypes.py).

### Searching on the NASNet search space

Please use the following scripts to use GDAS to search as in the original paper:
```
# search for both normal and reduction cells
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/NASNet-space-search-by-GDAS.sh cifar10 1 -1

# search for the normal cell while use a fixed reduction cell
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/NASNet-space-search-by-GDAS-FRC.sh cifar10 1 -1
```

**After searching**, if you want to re-train the searched architecture found by the above script, you can use the following script:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts/retrain-searched-net.sh cifar10 gdas-searched \
		     output/search-cell-darts/GDAS-cifar10-BN1/checkpoint/seed-945-basic.pth 96 -1
```
Note that `gdas-searched` is a string to indicate the name of the saved dir and `output/search-cell-darts/GDAS-cifar10-BN1/checkpoint/seed-945-basic.pth` is the file path that the searching algorithm generated.

The above script does not apply heavy augmentation to train the model, so the accuracy will be lower than the original paper.
If you want to change the default hyper-parameter for re-training, please have a look at `./scripts/retrain-searched-net.sh` and `configs/archs/NAS-*-none.config`.


### Searching on a small search space (NAS-Bench-201)

The GDAS searching codes on a small search space:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/GDAS.sh cifar10 1 -1
```

The baseline searching codes are DARTS:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/DARTS-V1.sh cifar10 1 -1
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/DARTS-V2.sh cifar10 1 -1
```

**After searching**, if you want to train the searched architecture found by the above scripts, please use the following codes:
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/NAS-Bench-201/train-a-net.sh '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|skip_connect~2|' 16 5
```
`|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|skip_connect~2|` represents the structure of a searched architecture. My codes will automatically print it during the searching procedure.


**Tensorflow codes for GDAS are in experimental state**, which locates at `exps-tf`.

# Citation

If you find that this project helps your research, please consider citing the following paper:
```
@inproceedings{dong2019search,
  title     = {Searching for A Robust Neural Architecture in Four GPU Hours},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1761--1770},
  year      = {2019}
}
```
