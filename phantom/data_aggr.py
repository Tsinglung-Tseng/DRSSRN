import tables
import numpy as np
from fast_progress import master_bar
from tables import IsDescription, Float32Col, Int8Col

fd = tables.open_file('D_1.h5')
fs = tables.open_file('S_1.h5')
fj = tables.open_file('J_1.h5')
F = tables.open_file('phantom.h5', 'w')


class Phantom(IsDescription):
    phantom = Float32Col([256, 256])
    grayscale = Float32Col(20)
    ptype = Int8Col(1)


def get_content(choice):
    if choice is fd:
        return fd.root.derenzo, 0, fd_counter.pop()
    elif choice is fs:
        return fs.root.sheppLogans, 1, fs_counter.pop()
    elif choice is fj:
        return fj.root.jaszczak, 2, fj_counter.pop()
    else:
        raise IndexError


table_phantom = F.create_table(F.root, 'phantom', Phantom, 'phantom')
row_phantom = table_phantom.row

choice = np.random.choice([fd,fs,fj])
fd_counter = list(range(80000))
fs_counter = list(range(80000))
fj_counter = list(range(80000))


for _ in master_bar(range(240000)):
    try:
        content, pt, i = get_content(choice)

        if content is fd.root.derenzo:
            row_phantom['phantom'] = content[i][0]
            row_phantom['grayscale'] = np.pad(content[i][1], (0, 20 - content[i][1].shape[0]), 'constant',
                                              constant_values=0)
            row_phantom['ptype'] = pt
            row_phantom.append()
        #             row_ptype.append()
        #             row_phantom.append()

        else:
            row_phantom['phantom'] = content[i][1]
            row_phantom['grayscale'] = np.pad(content[i][0], (0, 20 - content[i][0].shape[0]), 'constant',
                                              constant_values=0)
            row_phantom['ptype'] = pt
            row_phantom.append()
    #             row_ptype.append()
    #             row_phantom.append()

    except IndexError:
        pass

fd.close()
fs.close()
fj.close()
F.close()