# Single RGB-D Fitting: Total Human Modeling with an RGB-D Shot

This software demos the algorithm in the paper [Single RGB-D Fitting: Total Human Modeling with 
an RGB-D Shot](https://fangxianyong.github.io/home/papers/VRST19SIngleRGBD.pdf) by Xianyong Fang, et al. in VRST 2019.

## How to use
The general process of this demo is as follows:

1. Use the Kinect camera and its associated SDK to capture and save the RGB-D image, predicted joint points, and face
    orientation data;

2. Use LCR-net (https://thoth.inrialpes.fr/src/LCR-Net/) to predict human joint points from the color image;

3. Use ColorHandPose3D (https://github.com/lmb-freiburg/hand3d) to obtain the joint points of the human hand from the color image;

4. Use CRFasRNN (http://www.robots.ox.ac.uk/~szheng/CRFasRNN.html) to obtain a binary image for human segmentation 
    from the color image;

5. Use "PointCloudProcess" to get high-quality smooth point clouds;

6. Use "JointProcess" to the obtian the compound skeleton keypoints;

7. Take the data obtained above as input to“Fitting_Method” and then compute the final human model.

## Citation
If you find this repository and its associated data (e.g. images) is usefule for your work, please cite the paper, Thanks.

Xianyong Fang, Jikui Yang, Jie Rao, Linbo Wang and Zhigang Deng. [Single RGB-D Fitting: Total Human Modeling with 
an RGB-D Shot](https://fangxianyong.github.io/home/papers/VRST19SIngleRGBD.pdf), ACM Symposium on Virtual Reality Software and Technology (VRST 2019), pp. 24:1-24:11, 2019

## Contacts
Should you have any question regarding this software and its corresponding results, please contact:
Xianyong Fang (fangxianyong@ahu.edu.cn)
Jikui Yang (e16301112@stu.ahu.edu.cn)
Jie Rao (e18301094@stu.ahu.edu.cn)
Linbo Wang（wanglb@ahu.edu.cn)
Zhigang Deng (zdeng4@uh.edu)
