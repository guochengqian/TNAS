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
@inproceedings{qian2022meets,
  title={When NAS Meets Trees: An Efficient Algorithm for Neural Architecture Search},
  author={Qian, Guocheng and Zhang, Xuanyang and Li, Guohao and Zhao, Chen and Chen, Yukang and Zhang, Xiangyu and Ghanem, Bernard and Sun, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2782--2787},
  year={2022}
}
```

