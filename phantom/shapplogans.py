import numpy as np
from scipy.misc import imrotate
from .phantom import phantom
import cv2

# import sys
# sys.path
import matplotlib.pyplot as plt

def GenerateSheppLogans(Nimg, Nsize, sigma):
# GENERATESHEPPLOGANS Generaing multiple shapplogan phantoms.
# Suggest sigma = 0.1, plotwindow = [0.9 1.1]
    Em = np.zeros((10, 6))
    Em[:,0] = [1, -0.98, -0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    Em[:,1] = [0.69, 0.6624, 0.11, 0.16, 0.21, 0.046, 0.046, 0.046, 0.023, 0.023]
    Em[:,2] = [0.92, 0.874, 0.31, 0.41, 0.25, 0.046, 0.046, 0.023, 0.023, 0.046]
    Em[:,3] = [0, 0, .22, -.22, 0, 0, 0, -.08, 0, .06]
    Em[:,4] = [0, -.0184, 0, 0, .35, .1, -.1, -.605, -.605, -.605]
    Em[:,5] = [0, 0, -18, 18, 0, 0, 0, 0, 0, 0]
    sigmac = [0.1, 1, 1, 1, 1, 10]
    X = np.zeros((Nsize, Nsize, Nimg))
    for i in range(Nimg):
        E = np.random.randn(10,6)
        E = E*Em*sigma
        for j in range(6):
            E[:,j] = E[:, j] * sigmac[j]
        E = E + Em
        img = phantom(E, Nsize)
        img = imrotate(img, sigma*10*np.random.randn(),'nearest')
        # img = reshape(img, Nsize*Nsize, 1)
        X[:,:,i] = img
        # print(img.count())
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
    return X

# image = GenerateSheppLogans(9, 256, 0.1)
# # print(np.shape(image))
# plt.figure()
# plt.subplot(3,3,1)
# plt.imshow((image[:,:,0]))
# plt.subplot(3,3,2)
# plt.imshow((image[:,:,1]))
# plt.subplot(3,3,3)
# plt.imshow((image[:,:,2]),vmin=120,vmax=180)
# plt.subplot(3,3,4)
# plt.imshow((image[:,:,3]),vmin=120,vmax=180)
# plt.subplot(3,3,5)
# plt.imshow((image[:,:,4]),vmin=120,vmax=180)
# plt.subplot(3,3,6)
# plt.imshow((image[:,:,5]),vmin=120,vmax=180)
# plt.subplot(3,3,7)
# plt.imshow((image[:,:,6]),vmin=120,vmax=180)
# plt.subplot(3,3,8)
# plt.imshow((image[:,:,7]),vmin=120,vmax=180)
# plt.subplot(3,3,9)
# plt.imshow((image[:,:,8]),vmin=120,vmax=180)
# # plt.show()
# plt.show(block=True)