# DSANet: Dynamic Segment Aggregation Network for Video-Level Representation Learning (ACMMM 2021)
![1](DSANet.png)  

## Overview 
We release the code of the [DSANet](https://arxiv.org/abs/2105.12085) (Dynamic Segment Aggregation Network). We introduce the DSA module to capture relationship among snippets for video-level representation learning. Equipped with DSA modules, the top-1 accuracy of I3D ResNet-50 is improved to 78.2% on Kinetics-400. 

The core code to implement the Dynamic Segment Aggregation Module is `codes/models/modules_maker/DSA.py`. 

**[July 7, 2021]**  We release the core code of DSANet.

**[July 3, 2021]**  DSANet has been accepted by **ACMMM 2021**.

* [Prerequisites](#Prerequisites)
* [Data Preparation](#data-preparation)
* [Model Zoo](#model-zoo)
* [Testing](#testing)  
* [Training](#training)  



## Prerequisites

All dependencies can be installed using pip:

```sh
python -m pip install -r requirements.txt
```

Our experiments run on Python 3.7 and PyTorch 1.5. Other versions should work but are not tested.

## Download Pretrained Models

- Download ImageNet pre-trained models for offline environment

```sh
cd pretrained
sh download_imgnet.sh
```


- Download K400 pre-trained models for inference

TODO

<!-- Please refer to [Model Zoo](#model-zoo). -->


## Data Preparation
We follow the same data process with [MVFNet](https://github.com/whwu95/MVFNet/blob/main/data_process/DATASETS.md) for data preparation.



## Model Zoo


TODO




## Testing


```sh
bash dist_test_recognizer.sh CONFIG_PATH CHECKPOINT_PATH 8 
```

## Training
This implementation supports multi-gpu, `DistributedDataParallel` training, which is faster and simpler. 


- For example, to train DSANet with 8 gpus, you can run:

```sh
bash dist_train_recognizer.sh configs/kinetics/r50_e100_s1_8.py 8
```



## Acknowledgements
We especially thank the contributors of the [MVFNet](https://github.com/whwu95/MVFNet) and [mmaction](https://github.com/open-mmlab/mmaction) codebase for providing helpful code.


## License
This repository is released under the Apache-2.0. license as found in the [LICENSE](LICENSE) file.

## Related Work
[MVFNet](https://github.com/whwu95/MVFNet): Multi-View Fusion Network for Efficient Video Recognition, AAAI2021  [Paper](https://arxiv.org/pdf/2012.06977.pdf) | [Code](https://github.com/whwu95/MVFNet)


## Citation
If you think our work is useful, please feel free to cite our paper 😆 :
```
@inproceedings{wu2021dsanet,
  title={DSANet: Dynamic Segment Aggregation Network for Video-Level Representation Learning},
  author={Wu, Wenhao and Zhao, Yuxiang and Xu, Yanwu and Tan, Xiao and He, Dongliang and Zou, Zhikang and Ye, Jin and Li, Yingying and Yao, Mingde and Dong, Zichao and others},
  booktitle = {ACMMM},
  year={2021}
}
```





## Contact
For any question, please file an issue or contact
```
Wenhao Wu: wuwenhao17@mails.ucas.edu.cn
Yuxiang Zhao: yx.zhao@siat.ac.cn
```