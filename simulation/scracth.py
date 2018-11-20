import tables
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import glob
import os
import shutil
import pathlib
import subprocess
import time


# INDEX = 1

class KEYS:

    @classmethod
    def PHANTOM_SUB_DIR(cls, i):
        return f"derenzo_phantom_{i}"

    @classmethod
    def MAC_SUB(cls, i):
        return f"mac_sub{i}"

    PHANTON_SOURCE_DIR = "D_1.h5"
    PHANTOM_OUT_FILE = "phantomD.bin"
    WORKDIR = pathlib.Path('/mnt/gluster/qinglong/DLSimu')
    DEFAULT_MATERIAL_FILE = "range_material_phantomD.dat"
    DEFAULT_ACTIVITY_FILE = "activity_range_phantomD.dat"
    MACSET = pathlib.Path("/mnt/gluster/qinglong/macset")


FILE = tables.open_file(KEYS.PHANTON_SOURCE_DIR)


for INDEX in range(60, 100):
    derenzo, gray_scale = FILE.root.derenzo[INDEX]

    a = [0, gray_scale[0], gray_scale[1]]
    b = [0, gray_scale[0], gray_scale[1]]
    c = ['Air', 'Water', 'Air']

    range_material_phantomD1 = DataFrame([str(3)])
    range_material_phantomD2 = DataFrame([a, b, c]).T
    activity_range_phantomD = DataFrame([[2, gray_scale[0], gray_scale[1]],
                                         [np.nan, gray_scale[0], gray_scale[1]],
                                         [np.nan, int(10 * gray_scale[0]), int(10 * gray_scale[1])]]).T

    # workdir (/mnt/gluster/qinglong/DLSimu/derenzo_phantom_*)
    task_workdir = KEYS.WORKDIR / KEYS.PHANTOM_SUB_DIR(INDEX)

    # mkdir workdir
    try:
        os.makedirs(task_workdir)
        print(f"Directory {KEYS.PHANTOM_SUB_DIR(INDEX)} added.")
    except FileExistsError:
        print(f"Directory {KEYS.PHANTOM_SUB_DIR(INDEX)} exists.")

    # write phantom bin (derenzo_phantom_*/phantomD.bin)
    derenzo.tofile(str(task_workdir / KEYS.PHANTOM_OUT_FILE))

    # write range material phantom (derenzo_phantom_*/*.dat)
    with open(task_workdir / KEYS.DEFAULT_MATERIAL_FILE, "w") as f:
        range_material_phantomD1.to_csv(f, header=False, index=False, sep='\t', mode='a')
        range_material_phantomD2.to_csv(f, header=False, index=False, sep='\t', mode='a')
        print(f"Default material file wirten in {str(task_workdir/KEYS.DEFAULT_MATERIAL_FILE)}")

    # write activity range phantom
    with open(task_workdir / KEYS.DEFAULT_ACTIVITY_FILE, 'w') as f:
        activity_range_phantomD.to_csv(f,
                                       header=False,
                                       index=False,
                                       sep='\t',
                                       mode='a')
        print(f"Default material file wirten in {str(task_workdir/KEYS.DEFAULT_ACTIVITY_FILE)}")

    # make mac_sub
    for i in range(1, 5):
        mac_sub_dir = task_workdir / f'mac_sub{i}'
        try:
            os.mkdir(mac_sub_dir)
        except Exception as e:
            print(e)

        # copy "the rest"
        for p in (KEYS.MACSET / f'mac_sub{i}').iterdir():
            shutil.copyfile(p, (task_workdir / f'mac_sub{i}' / p.name))

        # overwritting main.mac
        shutil.copyfile((KEYS.MACSET / f"mac_sub{i}" / "main.mac"), (mac_sub_dir / "main.mac"))

        # overwritting dat and bin
        to_copy = [task_workdir / KEYS.DEFAULT_MATERIAL_FILE,
                   task_workdir / KEYS.DEFAULT_ACTIVITY_FILE,
                   task_workdir / KEYS.PHANTOM_OUT_FILE]
        for p in to_copy:
            shutil.copyfile(p, (mac_sub_dir / p.name))

    #     # add run script
    #     to_copy = [KEYS.MACSET/'cleanup.sh', KEYS.MACSET/"run_mac.sh"]
    #     for p in to_copy:
    #         shutil.copyfile(p, (task_workdir/p.name))

    # run
    for i in range(1, 5):
        mac_sub_dir = task_workdir / f'mac_sub{i}'
        subprocess.run(["pygate", "init", "subdir", "-n", "100"], cwd=str(mac_sub_dir))
        subprocess.run(["pygate", "init", "bcast"], cwd=str(mac_sub_dir))
        subprocess.run(["pygate", "submit"], cwd=str(mac_sub_dir))

    time.sleep(120)