import cv2
import pickle
import time

import numpy as np

from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans

from Utils import find_image_files


class VocabularyTree:
    def __init__(self, L=4, K=10):
        '''
        L - number of levels
        K - number of clusters at each level
        '''
        self.L = L
        self.K = K
        self.max_clusters = K ** L
        self.ca = None
        self.inverted_index = [{} for i in range(self.max_clusters)]

    def fit(self, data):
        self.ca = []
        self.ca.append(MiniBatchKMeans(n_clusters=self.K,
                                       batch_size=1000).fit(data))
        labels = self.ca[0].predict(data).astype(np.int32)

        for l in range(1, self.L):
            print("At level {}".format(l))
            ca_l = []
            n_c = self.K ** l
            for i in range(n_c):
                idx = np.where(labels == i)[0]

                if len(idx) > self.K * 3:
                    ca_l.append(MiniBatchKMeans(n_clusters=self.K,
                                                batch_size=1000).fit(data[idx, :]))
                    labels[idx] = labels[idx] * self.K + ca_l[i].predict(data[idx, :]).astype(np.int32)
                else:
                    ca_l.append(None)
                    labels[idx] = labels[idx] * self.K

            self.ca.append(ca_l)

        return self
        
    def build_inverted_index(self, data):
        pass
        
    def predict(self, data):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        c = self.ca[0].predict(data).astype(np.int32)

        for l in range(1, self.L):
            c_set = set(c)
            for i in c_set:
                idx = np.where(c == i)

                if self.ca[l][i] is None:
                    c[idx] *= (self.K ** (self.L - l))
                    break
                else:
                    c[idx] = c[idx] * self.K + self.ca[l][i].predict(data[idx[0], :])

        return c