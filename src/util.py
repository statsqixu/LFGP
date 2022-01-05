import numpy as np

def getdata1(n_tasks, n_workers, missing = 0.7):

    A1 = np.random.multivariate_normal(2 * np.ones(2, ), 3 * np.diag(np.ones(2, )), n_tasks // 2)
    A2 = np.random.multivariate_normal(-2 * np.ones(2, ), 3 * np.diag(np.ones(2, )), n_tasks // 2)

    A = np.r_[A1, A2]

    B1 = np.random.multivariate_normal(np.array([2, 1]), 3 * np.diag(np.ones(2, )), n_workers // 2)
    B2 = np.random.multivariate_normal(np.array([0, -2]), 3 * np.diag(np.ones(2, )), n_workers // 2)

    B = np.r_[B1, B2]

    R = A.dot(B.transpose())
    R = 1 / (1 + np.exp(-R))
    R[R >= 0.5] = 1
    R[R < 0.5] = 0

    task_mask = np.random.randint(low=0, high=n_tasks, size=np.int(n_tasks * n_workers * missing))
    worker_mask = np.random.randint(low=0, high=n_workers, size=np.int(n_tasks * n_workers * missing))

    R[task_mask, worker_mask] = np.nan

    row_idx, col_idx = np.where(~np.isnan(R))

    data = np.zeros((len(row_idx), 3))
    data[:, 0] = row_idx
    data[:, 1] = col_idx
    data[:, 2] = R[row_idx, col_idx]

    return data, A, B



