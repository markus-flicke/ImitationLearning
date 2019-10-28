import os
import numpy as np
DATA_FOLDER = '../dat/teacher'
idx = len(os.listdir(DATA_FOLDER))//2
observations = []
actions = []


for i in range(idx):
    if i % max(1, int(idx / 10)) == 0:
        print("preloading dat %d/%d" % (i, idx - 1))
    observations.append(np.load(os.path.join(DATA_FOLDER, "observation_%05d.npy" % i)))
    actions.append(np.load(os.path.join(DATA_FOLDER, "action_%05d.npy" % i)))


green_values = [204,229]
for i in range(len(observations)):
    for j in range(len(observations[i])):
        for k in range(len(observations[j])):
            colors = observations[i][j][k]
            observations[i][j][k] = [255,255,255] if colors[1] in green_values else colors


for i in range(idx):
    np.save(os.path.join(DATA_FOLDER, "../no_grass/observation_%05d.npy" % i), observations[i])

import shutil
for i in range(idx):
    shutil.copyfile(os.path.join(DATA_FOLDER, "action_%05d.npy" % i), os.path.join(DATA_FOLDER, "../no_grass/action_%05d.npy" % i))
shutil.copyfile(os.path.join(DATA_FOLDER, "count.npy"), os.path.join(DATA_FOLDER, "../no_grass/count"))