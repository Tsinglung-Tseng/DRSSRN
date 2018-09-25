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
# U(1,19) 20 for (1,19)
N = 20
rs = [R/(4*(i+tan(60))) for i in range(N)]

derenzo_section = {
    'key_point' : [(L/2-i*2*rs[i], 2*rs[i]) for i in range(N)],
    'ver_offset' : [2*rs[i] for i in range(N)],
    'hor_offset' : [2*rs[i]/tan(30) for i in range(N)]
}

# canvas = np.zeros([L, L])
#
# # U(0.0, 1.0)
# rand1 = np.random.random_sample()
# rand2 = np.random.random_sample()
# res = abs(rand1-rand2)
#
# # U(1, 19)
# denerzo_section_rs = np.random.choice(list(range(1,20)), 6)


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


def draw_section(derenzo_section, canvas, rs, L, R, ind, level=1, th=0):
    #     ind = 8
    v_0 = derenzo_section['ver_offset'][ind - 1]
    h_0 = derenzo_section['hor_offset'][ind - 1]

    f = 0
    h = L / 2 - R
    for i in list(reversed(list(range(ind)))):

        v = f * v_0
        for _ in range(i + 1):
            draw_dot(canvas,
                     rotate((derenzo_section['key_point'][ind - 1][0] + v,
                             derenzo_section['key_point'][ind - 1][1] + h), th),
                     rs[ind - 1],
                     level=level)
            v = v + 2 * v_0
        f += 1
        h += h_0


if __name__ == "__main__":
    import click


    # @click.command()
    # @click.option('--file')
    # def run(file):
    FILE_NAME = '/home/qinglong/node3share/derenzo/4.h5'
    f = tables.open_file(FILE_NAME, 'w')


    class Derenzo(IsDescription):
        derenzo = Float32Col([256, 256])
        value = Float32Col(2)


    # derenzo = f.create_group('/', 'derenzo')
    dd = f.create_table(f.root, 'derenzo', Derenzo, 'Derenzo')
    derenzo_row = dd.row

    for _ in tqdm(range(10000)):

        canvas = np.zeros([L, L])

        # U(0.0, 1.0)
        rand1 = np.random.random_sample()
        rand2 = np.random.random_sample()
        res = abs(rand1 - rand2)

        # U(1, 19)
        denerzo_section_rs = np.random.choice(list(range(1, 20)), 6)

        for i, n in zip(list(range(6)), denerzo_section_rs):
            draw_section(derenzo_section, canvas, rs, L, R, n, level=res, th=i*60)
        draw_dot(canvas, (L/2, L/2), 1.14*R, mode='append', level=min(rand1, rand2))

        # plt.imshow(block_reduce(canvas, (2, 2), func=np.max))
        # plt.show()
        derenzo_row['derenzo'] = block_reduce(canvas, (2, 2), func=np.max)
        derenzo_row['value'] = (max(rand1, rand2), min(rand1, rand2))
        derenzo_row.append()
    f.close()
