# TNAS


## Install 
```
git clone git@github.com:guochengqian/TNAS.git 
cd TNAS
source env_install.sh
```


## Search on NAS-Bench-201
```
CUDA_VISIBLE_DEVICES=0 python exps/NATS-algos/search-cell-tnas.py --cfg cfgs/search_cell/tnas.yaml 
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

