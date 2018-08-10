from random import shuffle
import glob
import numpy as np
import re
import scipy


def get_addrs(flickr_train_path):
    return glob.glob(flickr_train_path)

def get_index(addrs):
    index = [re.sub('\D', '',addr.split('/')[-1]) for addr in addrs]
    return index

def get_image(addr):
    from scipy import ndimage
    return np.array(ndimage.imread(addr, flatten=False)), re.sub('\D', '',addr.split('/')[-1])

def resize(image,num_px):
    import scipy
    resized_image = scipy.misc.imresize(image, size=(num_px,num_px))
    return resized_image

def crop_center(image,num_px):
    y,x,_ = image.shape
    startx = x//2-(num_px//2)
    starty = y//2-(num_px//2)

    if x<num_px or y<num_px:
        return None
    else:
        return image[starty:starty+num_px,startx:startx+num_px]

def write_image(addrs,preproc_out_path,prefix=''):
    for addr in addrs:
        image_array, index = get_image(addr)
        scipy.misc.imsave(prefix + preproc_out_path + str(index) + '.jpg', image_array)

def down_samp_image(image,block_size=(2,2,1),func=np.max):
    from skimage.measure import block_reduce
    return block_reduce(image, block_size, func)


def get_mini_batch(grp, shuffled_index):
    mini_batch = []
    for i in shuffled_index:
        mini_batch.append(grp[i])

    return np.asarray(mini_batch)