import os
import sys
sys.path.append('../task02_RANSAC')
from ransac_matching_sgd import *
import warnings
warnings.filterwarnings("ignore")


img = plt.imread("20180723165602.jpg")
h, w, c = img.shape
cv2_img = cv2.imread("20180723165602.jpg")
H, cv2_img_warp = random_warp(cv2_img)
img_warp = cv2.cvtColor(cv2_img_warp, cv2.COLOR_BGR2RGB)
print('H: ', H)
A = []
B = []
for i in range(15):
    x = random.randint(0, w - 1) * 1.
    y = random.randint(0, h - 1) * 1.
    A.append([x, y])
    src = np.array([x, y, 1]).reshape(3, )
    u, v, t = H.dot(src)
    B.append([u / t, v / t])

H_hat = ransacMatching(A, B)
print('Estimated H: ', H_hat)

B_hat = []
for p in A:
    x, y = p
    src = np.array([x, y, 1]).reshape(3, 1)
    u, v, t = H_hat.dot(src).ravel()
    B_hat.append([u / t, v / t])
print('B:', B)
print('B_hat:', B_hat)
plt.subplot(131)
plt.title("original")
for p in A:
    x, y = p
    plt.scatter(x, y, c='r')
plt.imshow(img)
plt.subplot(132)
for p in B:
    u, v = p
    plt.scatter(u, v, c='g')
plt.title("perspective transform")
plt.imshow(img_warp)
plt.subplot(133)
for p in B_hat:
    u, v = p
    plt.scatter(u, v, c='b')
plt.title("estimate perspective transform")
plt.imshow(img_warp)
plt.show()
