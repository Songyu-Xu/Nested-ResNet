<p align="center">

  <h1 align="center">Nested ResNet: A Vision-Based Method for Detecting the Sensing Area of a Drop-in Gamma Probe</h1>
  <div align="center">
    <h4><strong>Songyu Xu, Yicheng Hu, Jionglong Su, Daniel Elson, Baoru Huang*</strong></h4>
    <div align="center"></div>
  </div>
  <div align="center"></div>
</p>

![image](https://github.com/Songyu-Xu/Nested-ResNet/tree/main/figure/framework.png)

### Contents
1. [Requirements](#requirements)
2. [Dataset](#dataset)
3. [Training&Testing](#trainingtesting)
4. [Notes](#notes)

### Requirements
The code is implemented with `python=3.9` and `torch=2.2.1`
The required packages is listed in `requirements23.txt`
To install:
   - `conda create --name my_env python=3.9`
   - `conda activate my_env`
   - `pip install -r requirements.txt`

### Dataset
We use the Coffbea dataset provided in [Sensing_area_detection](https://github.com/br0202/Sensing_area_detection/tree/master). This is a dataset containing stereo laparascopic images of a gamma probe simulated working on
a silicon phantom, containing ground truth sensing location and depth map for evaluation. We appreciate authors for sharing the dataset.

We also use method from [unimatch](https://github.com/autonomousvision/unimatch) for estimating depth information. Please refer this paper for preparing the depth data. We appreciate authors for their paper and repository.

The structure of the dataset is as follows:
```
├──coffbea-2023
    ├──data/
        ├──camera0/
            ├──axis/
               ├──000000.txt
               .....
            ├──depthGT/
               ├──000000.jpg
               ├──000000.npy
               .....
            ├──rgb/
               ├──000000.jpg
               .....
        ├──camera1/
            ├──axis/
               ├──000000.txt
               .....
            ├──depthGT/
               ├──000000.jpg
               ├──000000.npy
               .....
            ├──rgb/
               ├──000000.jpg
               .....
        ├──deth_estimate/ 
            ├──00000_depth.pfm
            .....
```

### Training&Testing
1. Training:
	- `python main.py --mode train --model=stereo_skresnet_axis_depth`
	
2. Test:
    - `python main.py --mode test --model=stereo_skresnet_axis_depth`


### Acknowledgements
We thank the original authors of [Sensing_area_detection](https://github.com/br0202/Sensing_area_detection/tree/master) 
and [unimatch](https://github.com/autonomousvision/unimatch) for their excellent work.

