from matplotlib.backends.backend_agg import FigureCanvasAgg
from derenzo import DerenzoPhantom
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tables import IsDescription, Float32Col
import tables
import tqdm

FILENAME = '/home/qinglong/node3share/derenzo.h5'
f = tables.open_file(FILENAME, "w")

# def fig2data(fig):
#     fig.canvas.draw()
#     w, h = fig.canvas.get_width_height()
#     buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
#     buf.shape = (w, h, 4)
#     buf = np.roll(buf, 3, axis=2)
#     return buf
#
#
# def rgb2gray(rgb):
#     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


class Derenzo(IsDescription):
    derenzo = Float32Col([256,256])
    value   = Float32Col(2)

derenzo = f.create_group('/', 'derenzo')
dd = f.create_table(f.root.derenzo, 'derenzo', Derenzo, 'Derenzo')
derenzo_row = dd.row


for ind in range(100):
    # generating phantom
    radius = 37.0
    circles = np.random.choice((1, 2, 3, 4, 5, 7, 11), 6)

    my_phantom = DerenzoPhantom(radius, circles)

    agg = my_phantom.fig.canvas.switch_backends(FigureCanvasAgg)
    agg.draw()
    s, (width, height) = agg.print_to_buffer()

    gray = np.fromstring(s, np.uint8).reshape((height, width, 4))[:, :, 0]

    ct = Counter()
    real_derenzo = np.zeros([256, 256])
    rand1 = np.random.random_sample()
    rand2 = np.random.random_sample()

    real_derenzo[gray <= 80] = rand1
    real_derenzo[np.logical_and((gray > 80), (gray <= 200))] = rand2
    #     real_derenzo[pix > 200] = 0

    #     for i in range(255):
    #         for j in range(255):
    #             pix = gray[i][j]
    #             if pix > 200:
    #                 real_derenzo[i,j]=0
    #             elif pix <= 200 and pix > 80:
    #                 real_derenzo[i,j]=rand1
    #             else:
    #                 real_derenzo[i,j]=rand2

    derenzo_row['derenzo'] = real_derenzo
    derenzo_row['value'] = (rand1, rand2)
    derenzo_row.append()
    plt.close('all')

    if ind%50 == 0:
        dd.flush()
        print(f'Now : {ind}')
        plt.imshow(real_derenzo)
        plt.show()

f.close()