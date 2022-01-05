import numpy as np
from src.lfgp import baseLF, LF_MCGP, LF_fuse
from src.util import getdata1
import matplotlib.pyplot as plt


data, A, B = getdata1(100, 100)

label_true = np.array([1] * 50 + [0] * 50)

model1 = baseLF(mode="binary")
model1.fit(data, n_factors=4, verbose=1, learning_rate=1e-1, batch_size=512, weight_decay=0)
A1 = model1.model.task_factors.weight.detach().numpy()
B1 = model1.model.worker_factors.weight.detach().numpy()

model2 = LF_MCGP(mode="binary")
model2.fit(data, n_factors=4, n_task_group=2, n_worker_group=2, lambda1 = 0.0001, lambda2 = 0.0001, verbose=1, learning_rate=1e-1, batch_size=512, weight_decay=0)
A2 = model2.model.task_factors.weight.detach().numpy()
B2 = model2.model.worker_factors.weight.detach().numpy()

model3 = LF_fuse(mode="binary")
model3.fit(data, n_factors=2, verbose=1, learning_rate=1e-2, batch_size=16, weight_decay=0)
A3 = model3.model.task_factors.weight.detach().numpy()
B3 = model3.model.worker_factors.weight.detach().numpy()

fig, axes = plt.subplots(nrows=2, ncols=2)
cols = ["blue"] * 50 + ["red"] * 50
axes[0, 0].scatter(A[:, 0], A[:, 1], c=cols)
axes[0, 1].scatter(A1[:, 0], A1[:, 1], c=cols)
axes[1, 0].scatter(A2[:, 0], A2[:, 1], c=cols)
axes[1, 1].scatter(A3[:, 0], A3[:, 1], c=cols)
plt.setp(axes, xlim=(-6, 6), ylim=(-6, 6))
plt.show()
