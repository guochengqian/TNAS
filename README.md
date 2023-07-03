# TNAS
Official Pytorch Implementation of 
When NAS Meets Trees: An Efficient Algorithm for Neural Architecture Search

WARNING: this is not the final version. But a demo version. 
We are still working on this project in part time to push it for a conference or journal. After the paper gets accepted, the final version will be released. 

## Install 
```
git clone git@github.com:guochengqian/TNAS.git 
cd TNAS
source env_install.sh
```


## Search on NAS-Bench-201
```
CUDA_VISIBLE_DEVICES=0 python exps/NATS-algos/search-cell-tnas.py --cfg cfgs/search_cell/tnas.yaml

sbatch --array=0-4 --time=5:00:00 a100_tnas_alpha.sh cfgs/search_cell/tnas_warmup.yaml d_a=4
```


## Acknowledgment
This code is highly relied on [NATS-Bench](https://github.com/D-X-Y/AutoDL-Projects). 


## Citation
If you find that this project helps your research, please consider citing the related paper:
```
@article{dong2021nats,
  title   = {{NATS-Bench}: Benchmarking NAS Algorithms for Architecture Topology and Size},
  author  = {Dong, Xuanyi and Liu, Lu and Musial, Katarzyna and Gabrys, Bogdan},
  doi     = {10.1109/TPAMI.2021.3054824},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year    = {2021},
  note    = {\mbox{doi}:\url{10.1109/TPAMI.2021.3054824}}
}
```

