from matplotlib import pyplot as plt
import os
import shutil
import pickle

PATH = './answers'

for f_name in os.listdir(PATH):
    full_path = os.path.join(PATH, f_name)
    if os.path.isdir(full_path):
        continue
    with open(full_path, 'rb') as f:
        results = pickle.load(f)
    full_path = os.path.join(PATH, f_name.split('.')[0])
    os.mkdir(full_path)

    for i, res in enumerate(results):
        tmp_full_path = os.path.join(full_path, str(i+1))
        os.mkdir(tmp_full_path)

        for j, q in enumerate(res):
            os.mkdir(os.path.join(tmp_full_path, str(j+1)))
            plt.imsave(os.path.join(tmp_full_path, str(j+1),
                                    'query.jpg'), q[0])

            for k, s in enumerate(q[1]):
                shutil.copyfile(s, os.path.join(tmp_full_path, str(j+1),
                                                str(k)+'_'+os.path.split(s)[1]))
