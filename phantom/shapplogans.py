import numpy as np
import tables
import math
from scipy.misc import imrotate, bytescale
from scipy.misc import toimage, fromimage
from tables import IsDescription, Float32Col
import matplotlib.pyplot as plt
from tqdm import tqdm

#                          A    a       b      x0     y0   phi
shepp_logan = np.array([[  1,  .69,     .92,    0,     0,  0],
                        [-.98, .6624, .8740,    0,-.0184,  0],
                        [-.02, .1100, .3100,  .22,     0,-18],
                        [-.02, .1600, .4100, -.22,     0, 18],
                        [ .01, .2100, .2500,    0,   .35,  0],
                        [ .01, .0460, .0460,    0,    .1,  0],
                        [ .01, .0460, .0460,    0,   -.1,  0],
                        [ .01, .0460, .0230, -.08, -.605,  0],
                        [ .01, .0230, .0230,    0, -.606,  0],
                        [ .01, .0230, .0460,  .06, -.605,  0]])


modified_shepp_logan =np.array([[ 1,   .69,   .92,     0,     0,     0],
                                [-.8,  .6624,  .8740,   0,  -.0184,   0],
                                [-.2,  .1100,  .3100,  .22,    0,   -18],
                                [-.2,  .1600,  .4100, -.22,    0,    18],
                                [ .1,  .2100,  .2500,   0,   .35,     0],
                                [ .1,  .0460,  .0460,   0,    .1,     0],
                                [ .1,  .0460,  .0460,   0,   -.1,     0],
                                [ .1,  .0460,  .0230, -.08,  -.605,   0],
                                [ .1,  .0230,  .0230,   0,   -.606,   0],
                                [ .1,  .0230,  .0460,  .06,  -.605,   0]])



# def _imrotate(arr, mode, angle, interp='bilinear'):
#     """
#     Rotate an image counter-clockwise by angle degrees.
#     This function is only available if Python Imaging Library (PIL) is installed.
#     .. warning::
#         This function uses `bytescale` under the hood to rescale images to use
#         the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
#         It will also cast data for 2-D images to ``uint32`` for ``mode=None``
#         (which is the default).
#     Parameters
#     ----------
#     arr : ndarray
#         Input array of image to be rotated.
#     angle : float
#         The angle of rotation.
#     interp : str, optional
#         Interpolation
#         - 'nearest' :  for nearest neighbor
#         - 'bilinear' : for bilinear
#         - 'lanczos' : for lanczos
#         - 'cubic' : for bicubic
#         - 'bicubic' : for bicubic
#     Returns
#     -------
#     imrotate : ndarray
#         The rotated array of image.
#     """
#     arr = np.asarray(arr)
#     func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
#     im = toimage(arr, mode = mode)
#     im = im.rotate(angle, resample=func[interp])
#     return fromimage(im)


def phantom(*kargs):
    ellipse, n = parse_inputs(*kargs)
    p = np.zeros((n, n))
    xax = np.array([(i-(n-1)/2) / ((n-1)/2) for i in range(n)])
    xg = np.tile(xax, (n,1))
    for k in range(len(ellipse)):
        asq = ellipse[k,1]**2
        bsq = ellipse[k,2]**2
        phi = ellipse[k,5]*math.pi / 180
        x0 = ellipse[k,3]
        y0 = ellipse[k,4]
        A = ellipse[k,0]
        x = xg - x0
        y = xg.T[::-1] - y0
        # y = xg.T - y0
        cosp = math.cos(phi)
        sinp = math.sin(phi)
        con = (x*cosp + y*sinp)**2 / asq + (y*cosp + x*sinp)**2 / bsq
        index_x, index_y = np.where(con <= 1)
        # print(index)
        for idx_x, idx_y in zip(index_x, index_y):
            p[idx_x, idx_y] = p[idx_x, idx_y] + A
    # print(p[:,100])
    return p


def parse_inputs(*kargs):
    n = 256
    e = []
    defaults = ['shepp-logan', 'modified shepp-logan']
    for arg in kargs:
        if isinstance(arg, str):
            arg = arg.lower()
            idx = defaults.index(arg)
            if not idx:
                raise IOError
            elif idx == 0:
                e = shepp_logan
            elif idx == 1:
                e = modified_shepp_logan
        elif isinstance (arg, int):
            n = arg
        elif (arg.ndim == 2) and (arg.shape[1] == 6):
            e = arg
        else:
            raise IOError

    if not len(e):
        e = modified_shepp_logan
    return e, n


def GenerateSheppLogans(Nimg, Nsize, sigma):
    """
    GENERATESHEPPLOGANS Generaing multiple shapplogan phantoms.
    Suggest sigma = 0.1, plotwindow = [0.9 1.1]
    """

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
        X[:,:,i] = img
    X2list = img.flatten().tolist()
    intensity = list(set(X2list))
    intensity.sort(key = X2list.index)
    print(intensity)
    return X, intensity

#the first column of E is indensity
phantom1, intensity = GenerateSheppLogans(1, 256, 0.1)
print(intensity)

plt.imshow(np.reshape(phantom1, (256,256)))
plt.show()

FILE_NAME = '/home/qinglong/node3share/derenzo/S_1.h5'
f = tables.open_file(FILE_NAME, 'w')


class SheppLogans(IsDescription):
    sheppLogans = Float32Col([256, 256])
    inten = Float32Col(20)


dd = f.create_table(f.root, 'sheppLogans', SheppLogans, 'sheppLogans')
jaszczak_row = dd.row

for _ in tqdm(range(100000)):
    phantom1, inten = GenerateSheppLogans(1, 256, 0.1)

    jaszczak_row['sheppLogans'] = np.reshape(phantom1, (256, 256))
    jaszczak_row['inten'] = np.pad(inten, (0, 20 - len(inten)), 'constant', constant_values=-1)
    jaszczak_row.append()