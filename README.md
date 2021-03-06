# MA-CDLS


## Introduction
MA-CDLS is a memetic algorithm based on community detection for latency-sensitive and energy-aware service migration optimization in 5G mobile edge computing.

## Prerequisites:
python 3.8.2  
numpy 1.18.2  
pandas 1.0.3  
scipy 1.4.1  
pyyaml 5.3.1  
xlrd 1.2.0  
matplotlib 3.2.1  

## Usage
1.  High-performance mode
```
python main.py globalCfg_single_obj.yml  
```
2.  Energy-efficient mode
```
python main.py globalCfg_multi_obj.yml  
```

## Simulative Network
<img src="data/SN2/Slot%201.png" width="800" /><br/>

Our simulation experiment is based on the Wireless Network Simulator version 2.0: Z. Becvar, J. Kim, I. Kim, E. D. Santis, and J. Vidal, “6G in the sky: On demand intelligence at the edge of 3D networks (invited paper),” ETRI Journal, vol. 42, no. 5, pp. 643–656, 2020.

## Acknowledgments
* Please refer to the original paper: G. Li, L. Liu, Z. Liang, X. Ma, Z. Zhu, "Memetic Algorithm Based on Community Detection for Energy-Efficient Service Migration Optimization in 5G Mobile Edge Computing", in 32st IEEE Annual International Symposium on Personal, Indoor and Mobile Radio Communications, PIMRC 2021, Helsinki, Finland, 13-16 September, 2021. (Accept)


```
@inproceedings{LiG21Memetic,  
  author    = {Guo Li and  
               Ling Liu and  
               Zhengping Liang and  
               Xiaoliang Ma and  
               Zexuan Zhu},  
  title     = {Memetic Algorithm Based on Community Detection for Energy-Efficient Service Migration Optimization in 5G Mobile Edge Computing},  
  booktitle = {32st {IEEE} Annual International Symposium on Personal, Indoor and
               Mobile Radio Communications, {PIMRC} 2021, Helsinki, Finland,
               13-16 September, 2021},  
  pages     = {1--7},  
  publisher = {{IEEE}},  
  year      = {2021}  
}  
```

## PIMRC 2021 websites
<a href="https://pimrc2021.ieee-pimrc.org/" target="_blank">IEEE PIMRC 2021</a>

<a href="https://www.oulu.fi/6gflagship/node/209031" target="_blank">6G Flagship PIMRC 2021</a>