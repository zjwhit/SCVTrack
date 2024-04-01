# SCVTrack
## Robust-3D-Tracking-with-Quality-Aware-Shape-Completion
This repository contains the implementation of our method for 3D single object tracking with shape completion. Our approach focuses on constructing precise shape representations using dense and complete point clouds, achieved through shape completion techniques. The provided code includes a voxelized 3D tracking framework with a quality-aware shape completion mechanism, as well as modules for relation modeling. 

https://doi.org/10.1609/aaai.v38i7.28544

## Setup

Installation

+ Install pytorch

  ```
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
  ```

+ Install other dependencies:

  ```
  pip install -r requirement.txt
  ```

## Training & Testing

To train a model, you must specify the `.yaml` file with `--cfg` argument. The `.yaml` file contains all the configurations of the dataset and the model. We provide `.yaml` files under the [*cfgs*](./cfgs) directory.

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py  --cfg cfgs/cfg.yaml  --batch_size 64 --epoch 60 --preloading
```
To test a trained model, specify the checkpoint location with `--checkpoint` argument and send the `--test` flag to the command.

```bash
python main.py  --cfg cfgs/cfg.yaml  --checkpoint /path/to/checkpoint/xxx.ckpt --test
```

## 


## Acknowledgment
This repo is built upon [M2 Track](https://github.com/Ghostish/Open3DSOT).
