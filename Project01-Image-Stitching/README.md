# Project 01 | Image Stitching
---
### 1. Find keypoints and descriptors in each image
[key_descriptor.py](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/src/key_descriptor.py): SIFT  
`from key_descriptor import SIFT `  
`sift = SIFT()`  
`kp1, des1 = sift.detect_and_compute(img1, True, file_name)`  
![SIFT Keypoints](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/result/Nikki_garden_01_kp.jpg?raw=true)

### 2. Estimate the homography matrix of matched keypoints
(1) Get coordinates of matched keypoints  
[key_descriptor.py](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/src/key_descriptor.py): 
find matched keypoints where the correlation matrix of descriptors is greater than a threshold   
Draw lines between matched keypoints:  
`sift.draw_spindle(img1, img2, p1, p2, lwd=2, draw_samples=50)`  
![Matching Spindle](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/result/Nikki_garden_match_sample50.jpg?raw=true)

(2) RANSAC matching: calculate the homography matrix  
[ransac_matching.py](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/src/ransac_matching.py):
 return the homography matrix in 'numpy.ndarray'.

### 3. Image stitching
[image_stitching.py](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/src/image_stitching.py):  
`stitch = image_stitching()`  
`img_stitched = stitch.get_img_stitched(img1, img2, p1, p2, H_12)`  
`cv2.imshow('image stitching', img_stitched)`  
![Stitched image](https://github.com/GGGHSL/Deep-Learning-CV-master/blob/master/Project01-Image-Stitching/result/Nikki_garden_stitched_filled.jpg?raw=true)