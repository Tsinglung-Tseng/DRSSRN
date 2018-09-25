import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from tables import IsDescription, Float32Col
import tables
from tqdm import tqdm


def cos(deg):
    return np.cos(-1 * deg * (np.pi / 180))


def sin(deg):
    return np.sin(deg * (np.pi / 180))


def tan(deg):
    return np.tan(deg * (np.pi / 180))


L = 512
R = 210
R_0 = 128


def rotate(p, th):
    """
    usage: rotate(P, 60)
    """
    p0 = (L / 2, L / 2)

    return ((p[0] - p0[0]) * cos(th) - (p[1] - p0[1]) * sin(th) + p0[0],
            (p[0] - p0[0]) * sin(th) + (p[1] - p0[1]) * cos(th) + p0[1])


def draw_dot(canvas, p, r, mode='fix', level=1):
    x, y = np.meshgrid(range(canvas.shape[0]), range(canvas.shape[1]))
    mask = (x - p[0])**2 + (y - p[1])**2 <= r**2
    if mode == 'fix':
        canvas[mask] = level
    else:
        canvas[mask] += level


def sample(seed):
    """
    useage: num_of_samples, minimum_sample, samples = sample()
    """
    samples = np.random.random_sample(seed)
    min_s = min(samples)
    samples=np.delete(samples, np.where(samples==min_s))
    return len(samples),min_s, samples-min_s


FILE_NAME = '/home/qinglong/node3share/derenzo/J_0.h5'
f = tables.open_file(FILE_NAME, 'w')


class Jaszczak(IsDescription):
    jaszczak = Float32Col([256, 256])
    radius = Float32Col(10)
    inten = Float32Col(10)


dd = f.create_table(f.root, 'jaszczak', Jaszczak, 'jaszczak')
jaszczak_row = dd.row


for _ in tqdm(range(10000)):
    # canvas
    canvas = np.zeros([L, L])

    # aligned circle radius and intensity with seed
    seed = np.random.choice(range(7, 12))
    num_circles, _, rs_temp = sample(seed)
    _, base_inten, inten = sample(seed)

    r_max = R_0 * sin(180 / num_circles) * 0.9
    rs = rs_temp * (r_max / max(rs_temp))

    # draw content
    for i, r, inte in zip(list(range(num_circles)), rs, inten):
        draw_dot(canvas,
                 rotate((L / 2, R_0), i * 360 / num_circles),
                 r, mode='fix', level=inte)

    # draw background
    draw_dot(canvas, (L / 2, L / 2), R, mode='add', level=base_inten)

    #     plt.imshow(block_reduce(canvas, (2, 2), func=np.max))
    #     plt.show()
    jaszczak_row['jaszczak'] = block_reduce(canvas, (2, 2), func=np.max)
    jaszczak_row['radius'] = np.pad(rs, (0, 10 - rs.shape[0]), 'constant', constant_values=0)
    jaszczak_row['inten'] = np.pad(inten + base_inten, (0, 10 - rs.shape[0]), 'constant', constant_values=0)
    jaszczak_row.append()
f.close()