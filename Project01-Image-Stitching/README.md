# Project 01 | Image Stitching

## Run testcase:
```bash
cd ~/Deep-Learning-CV-master/Project01-Image-Stitching
python test.py
```
## Source code:
### 1. Find keypoints and descriptors in each image
[key_descriptor.py](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/src/key_descriptor.py): SIFT  
```python
import os
import sys
import numpy as np
import cv2
sys.path.append('./src')
from key_descriptor import SIFT
from ransac_matching import RansacMatching
from image_stitching import image_stitching

filepath1 = "./image/Nikki_garden_01.jpg"
filepath2 = "./image/Nikki_garden_03.jpg"
img1 = cv2.imread(filepath1)
img2 = cv2.imread(filepath2)
 
sift = SIFT()
kp1, des1 = sift.detect_and_compute(img1, False)
kp2, des2 = sift.detect_and_compute(img2, False)
``` 
![SIFT Keypoints](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/result/Nikki_garden_01_kp.jpg?raw=true)

### 2. Estimate the homography matrix of matched keypoints
#### (1) Get coordinates of matched keypoints  
[key_descriptor.py](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/src/key_descriptor.py): 
find matched keypoints where the correlation matrix of descriptors is greater than a threshold.   
Draw lines between matched keypoints:  
```python
p1, p2 = sift.get_matching_points(kp1, des1, kp2, des2, 0.995)
sift.draw_spindle(img1, img2, p1, p2, lwd=2)
sift.draw_spindle(img1, img2, p1, p2, lwd=2, draw_samples=50)
```  
![Matching Spindle](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/result/Nikki_garden_match_sample50.jpg?raw=true)

#### (2) RANSAC matching: calculate the homography matrix  
[ransac_matching.py](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/src/ransac_matching.py):
 return the homography matrix in 'numpy.ndarray'.
```python
ransac = RansacMatching()
H_12 = ransac.get_homography(p1, p2, p1.shape[0] // 5)  # H_12 * image2 -> image1
```

### 3. Image stitching
[image_stitching.py](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/src/image_stitching.py):  
```python
stitch = image_stitching()
img_stitched = stitch.get_img_stitched(img1, img2, p1, p2, H_12)
cv2.imshow('image stitching', img_stitched)
```  
![Stitched image](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/result/Nikki_garden_stitched_filled.jpg?raw=true)