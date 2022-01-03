import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from src.model import baseMF, MCGP_MF, fuse_MF
from src.container import LFGPDataset

import numpy as np

def _return_device(device):
    
    if device == "cpu":

        return "cpu"

    elif device == "gpu":

        return "cuda:0"

    elif device == "default":

        if torch.cuda.is_available():
            return "cuda:0"

        else:
            return "cpu"

class baseLF():

    def __init__(self, mode="binary"):

        self.mode = mode

    def fit(self, data, n_factors=3, epochs=100, learning_rate=1e-4, verbose=0, opt_func=Adam, weight_decay=1e-2, 
                    batch_size=128, device="default"):

        _device = _return_device(device)

        if verbose > 0:

            print("--------- The program is running on {} ---------".format(_device))

        self.device = _device

        n_tasks = len(np.unique(data[:, 0]))
        n_workers = len(np.unique(data[:, 1]))

        if self.mode == "binary":

            self.model = baseMF(n_tasks, n_workers, n_factors, loss="BCE")

        elif self.mode == "continuous":

            self.model = baseMF(n_tasks, n_workers, n_factors, loss="MSE")

        self.model = self.model.to(self.device)

        task_tsr = torch.from_numpy(data[:, 0]).long()
        worker_tsr = torch.from_numpy(data[:, 1]).long()
        label_tsr = torch.from_numpy(data[:, 2]).float()

        dataset = LFGPDataset(task_tsr, worker_tsr, label_tsr)
        loader = DataLoader(dataset, batch_size)

        if verbose == 0:
            print_history = False
        elif verbose == 1:
            print_history = True
          
        history = []
        history += self.model.fit(epochs=epochs, learning_rate=learning_rate,
                                train_loader=loader, print_history=print_history,
                                opt_func=opt_func, weight_decay=weight_decay, device=self.device)

        return history

class LF_MCGP():

    def __init__(self, mode="binary"):

        self.mode = mode

    def fit(self, data, n_factors=3, n_task_group=2, n_worker_group=2, 
                    lambda1=1e-2, lambda2=1e-2, epochs=100,
                    learning_rate=1e-4, verbose=0, opt_func=Adam, weight_decay=1e-2,
                    batch_size=128, device="default"):

        _device = _return_device(device)

        if verbose > 0:

            print("--------- The program is running on {} ---------".format(_device))

        self.device = _device

        n_tasks = len(np.unique(data[:, 0]))
        n_workers = len(np.unique(data[:, 1]))

        if self.mode == "binary":

            self.model = MCGP_MF(n_tasks, n_workers, n_factors, 
                                n_task_group, n_worker_group,
                                lambda1, lambda2, loss="BCE")

        elif self.mode == "continuous":

            self.model = MCGP_MF(n_tasks, n_workers, n_factors, 
                                n_task_group, n_worker_group,
                                lambda1, lambda2, loss="MSE")

        self.model = self.model.to(self.device)

        task_tsr = torch.from_numpy(data[:, 0]).long()
        worker_tsr = torch.from_numpy(data[:, 1]).long()
        label_tsr = torch.from_numpy(data[:, 2]).float()

        dataset = LFGPDataset(task_tsr, worker_tsr, label_tsr)
        loader = DataLoader(dataset, batch_size)

        if verbose == 0:
            print_history = False
        elif verbose == 1:
            print_history = True

        history = []
        history += self.model.fit(epochs=epochs, learning_rate=learning_rate,
                                    train_loader=loader, print_history=print_history,
                                    opt_func=opt_func, weight_decay=weight_decay, device=self.device)

        return history

class LF_fuse():

    def __init__(self, mode="binary"):
    
        self.mode = mode

    def fit(self, data, n_factors=3, lambda11=1e-2, lambda12=1e-2, 
                    lambda21=1e-2, lambda22=1e-2, epochs=100,
                    learning_rate=1e-4, verbose=0, opt_func=Adam, weight_decay=1e-2,
                    batch_size=128, device="default"):

        _device = _return_device(device)

        if verbose > 0:

            print("--------- The program is running on {} ---------".format(_device))

        self.device = _device

        n_tasks = len(np.unique(data[:, 0]))
        n_workers = len(np.unique(data[:, 1]))

        if self.mode == "binary":

            self.model = fuse_MF(n_tasks, n_workers, n_factors, 
                                lambda11, lambda12, lambda21, lambda22, loss="BCE")

        elif self.mode == "continuous":

            self.model = fuse_MF(n_tasks, n_workers, n_factors, 
                                lambda11, lambda12, lambda21, lambda22, loss="MSE")

        self.model = self.model.to(self.device)

        task_tsr = torch.from_numpy(data[:, 0]).long()
        worker_tsr = torch.from_numpy(data[:, 1]).long()
        label_tsr = torch.from_numpy(data[:, 2]).float()

        dataset = LFGPDataset(task_tsr, worker_tsr, label_tsr)
        loader = DataLoader(dataset, batch_size)

        if verbose == 0:
            print_history = False
        elif verbose == 1:
            print_history = True

        history = []
        history += self.model.fit(epochs=epochs, learning_rate=learning_rate,
                                    train_loader=loader, print_history=print_history,
                                    opt_func=opt_func, weight_decay=weight_decay, device=self.device)

        return history