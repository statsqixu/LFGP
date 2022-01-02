import numpy as np
from src.lfgp import baseLF, LF_MCGP


data = np.loadtxt("rte_crowd.txt")
data = data - 1 # index and label starting from 0
label_true = np.loadtxt("rte_truth.txt")
label_true = label_true[:, 1] - 1

model1 = baseLF(mode="binary")
model1.fit(data, n_factors=4, verbose=1, learning_rate=1e-1, batch_size=512, weight_decay=0)

model2 = LF_MCGP(mode="binary")
model2.fit(data, n_factors=4, verbose=1, learning_rate=1e-1, batch_size=512, weight_decay=0)
