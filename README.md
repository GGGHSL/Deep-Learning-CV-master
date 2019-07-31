# Deep-Learning-CV
>用于提交“深度学习与计算机视觉”课程的作业。

>💗友情出镜💗: [**@暖暖-Nikki**](https://weibo.com/u/6775494073?is_all=1)  
![@暖暖-Nikki](https://wx4.sinaimg.cn/mw690/007oxhwtgy1g1t1yc3q1nj31dq0rse82.jpg "请把我的code带回你的家~请把你的STAR留下~~")

## Week 01 | CV Fundamental I
### Image Processing (Low Level)
>**Data Augmentation:**
>* Gamma correction
>* Similarity Transformation (Scale + Rotation + Translation)
>* Affine Transformation
>* Perspective Transformation

## Week 02 | CV Fundamental II
### 1. Image Processing (Low Level)
>**Image Convolution:**
>* First-order & Second-order Derivative
>* Gaussian Kernel
>* Image Sharpen: Laplacian Operator
>* Edge Detection: Sober Operator
>* Image Blurring: Median & Gaussian
### 2. Feature Point (Mid Level)
**(1) What is Feature Point?**  

**(2) What is a good Feature Point?**
>**Very informational:**
>* Harris Corner Detector  

>**Rotation & Brightness resistance**

>**Scale resistance**
>* Corner point is not satisfied.

**(3) What is the form of Feature Point?**
>* Physical in location;
>* Abstract in formation.  
>===> **Feature Descriptor** 
     

**(4) How to get a Feature Descriptor?**
>**SIFT**
>* Generate scale-space: DoG
>* Scale-space Extrema Detection
>* Accurate key-point localization
>* Eliminating edge responses
>* Orientation assignment
>* Key-point descriptor

### 3. Classical CV Procedure (High Level)
>**Image Stitching**  
[Project01 | Image Stitching](https://github.com/GGGHSL/Deep-Learning-CV-master/tree/master/Project01-Image-Stitching): Updated

>**Image Classification (Bow / ML)**

## Week 03 | ML Fundamental I
>* Introduction To Machine Learning  
>* Classical Supervised Learning  
>   * Linear Regression & Logistic Regression
>       * SGD
>       * Zigzag?

## Week 04 | ML Fundamental II
>* Classical Supervised Learning
>   * Neural Network
>   * Back Propagation
>   * Regularization

## Week 05 | ML Fundamental III
>* Classical Supervised Learning
>   * SVM
>       * Linear SVM
>       * Kernels

>* Classical Unsupervised Learning
>   * K-Means & K-Means++

>* Concept & Problems
>   * Under-fitted & Over-fitted
>   * Bias & Variance
>   * ...