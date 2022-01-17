import numpy as np
import torch
from fast_pytorch_kmeans import KMeans

class baseMF(torch.nn.Module):

    """
    Matrix factorization for binary/continuous entries with L_2 loss
    """

    def __init__(self, n_tasks, n_workers, n_factors=3, loss="BCE"):

        super().__init__()

        self.n_tasks = n_tasks
        self.n_workers = n_workers
        self.n_factors = n_factors
        self.loss_func = loss
        self.task_factors = torch.nn.Embedding(n_tasks, n_factors)
        self.worker_factors = torch.nn.Embedding(n_workers, n_factors)

    def forward(self, task, worker):

        if self.loss_func == "BCE":

            score = torch.sigmoid((self.task_factors(task) * self.worker_factors(worker)).sum(1))

        elif self.loss_func == "MSE":

            score = (self.task_factors(task) * self.worker_factors(worker)).sum(1)

        return score

    def training_step(self, batch):

        _, tasks, workers, labels = batch

        output = self(tasks, workers)

        if self.loss_func == "BCE":
            loss_func = torch.nn.BCELoss()
        elif self.loss_func == "MSE":
            loss_func = torch.nn.MSELoss()

        loss = loss_func(output, labels)

        return loss

    def epoch_end(self, epoch, result):

        print("Epoch: {} - Training loss: {:.4f}".format(epoch, result))

    def fit(self, epochs, learning_rate, train_loader, print_history, opt_func, weight_decay, device):
    
        history = []
        task_optimizer = opt_func(self.task_factors.parameters(), learning_rate, weight_decay=weight_decay)
        worker_optimizer = opt_func(self.worker_factors.parameters(), learning_rate, weight_decay=weight_decay)
        
        task_optimizer.zero_grad()
        worker_optimizer.zero_grad()

        task_scheduler = torch.optim.lr_scheduler.ExponentialLR(task_optimizer, gamma=0.999)
        worker_scheduler = torch.optim.lr_scheduler.ExponentialLR(worker_optimizer, gamma=0.999)

        for epoch in range(epochs):
            # training
            for batch in train_loader:
                batch = [item.to(device) for item in batch]
                loss = self.training_step(batch)
                loss.backward()
                
                task_optimizer.step()
                task_optimizer.zero_grad()
                task_scheduler.step()

                worker_optimizer.step()
                worker_optimizer.zero_grad()
                worker_scheduler.step()
                
            result = self._evaluate(train_loader, device)
            if print_history:
                self.epoch_end(epoch, result)
            history.append(result)
            
        return history

    def _evaluate(self, train_loader, device):
        
        outputs = []
        for batch in train_loader:
            batch = [item.to(device) for item in batch]
            outputs.append(self.training_step(batch))
        
        return torch.stack(outputs).mean()


class MCGP_MF(torch.nn.Module):

    """
    Matrix factorization for binary/continuous entries with Multi-Centroid Grouping Penalty
    """

    def __init__(self, n_tasks, n_workers, n_factors=3, 
                    n_task_group=2, n_worker_group=2, lambda1=1e-2, lambda2=1e-2, loss="BCE"):

        super().__init__()

        self.n_tasks = n_tasks
        self.n_workers = n_workers
        self.n_factors = n_factors
        self.n_task_group = n_task_group
        self.n_worker_group = n_worker_group
        self.loss_func = loss
        self.task_factors = torch.nn.Embedding(n_tasks, n_factors)
        self.worker_factors = torch.nn.Embedding(n_workers, n_factors)
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        if self.lambda1 > 0:
            self.P = torch.randint(high=n_task_group, size=(n_tasks, ))     # task group membership
        else:
            self.P = torch.ones((n_tasks, ))
            self.n_task_group = 1

        if self.lambda2 > 0:
            self.Q = torch.randint(high=n_worker_group, size=(n_workers, )) # worker group membership
        else:
            self.Q = torch.ones((n_workers, ))
            self.n_worker_group = 1

    def forward(self, task, worker):

        if self.loss_func == "BCE":
    
            score = torch.sigmoid((self.task_factors(task) * self.worker_factors(worker)).sum(1))

        elif self.loss_func == "MSE":

            score = (self.task_factors(task) * self.worker_factors(worker)).sum(1)

        return score

    def MCGP(self, task_factors, worker_factors, P, Q):

        task_centroid = torch.zeros(self.n_task_group, self.n_factors)

        penalty1 = 0

        if self.lambda1 > 0:

            for i in range(self.n_task_group):

                task_centroid[i, ] = torch.mean(task_factors[P == i, ], dim=0)

                penalty1 += torch.sum(torch.square(task_factors[P == i, ] - task_centroid[i, ]))

            penalty1 = self.lambda1 * penalty1

        worker_centroid = torch.zeros(self.n_worker_group, self.n_factors)

        penalty2 = 0
        
        if self.lambda2 > 0:

            for i in range(self.n_worker_group):

                worker_centroid[i, ] = torch.mean(worker_factors[Q == i, ], dim=0)

                penalty2 += torch.sum(torch.square(worker_factors[Q == i, ] - worker_centroid[i, ]))

            penalty2 = self.lambda2 * penalty2

        penalty = penalty1 + penalty2

        return penalty


    def training_step(self, batch):

        _, tasks, workers, labels = batch

        output = self(tasks, workers)

        if self.loss_func == "BCE":
            loss_func = torch.nn.BCELoss()
        elif self.loss_func == "MSE":
            loss_func = torch.nn.MSELoss()

        loss = loss_func(output, labels)

        penalty = self.MCGP(self.task_factors(tasks),
                            self.worker_factors(workers),
                            self.P[tasks], self.Q[workers])

        return loss + penalty

    def epoch_end(self, epoch, result):

        print("Epoch: {} - Training loss: {:.4f}".format(epoch, result))

    def _evaluate(self, train_loader, device):
            
        outputs = []
        for batch in train_loader:
            batch = [item.to(device) for item in batch]
            outputs.append(self.training_step(batch))
        
        return torch.stack(outputs).mean()

    def fit(self, epochs, learning_rate, train_loader, print_history, opt_func,
                weight_decay, device):

        history = []
        task_optimizer = opt_func(self.task_factors.parameters(), learning_rate, weight_decay=weight_decay)
        worker_optimizer = opt_func(self.worker_factors.parameters(), learning_rate, weight_decay=weight_decay)

        task_optimizer.zero_grad()
        worker_optimizer.zero_grad()

        task_kmeans = KMeans(self.n_task_group, verbose=0)
        worker_kmeans = KMeans(self.n_worker_group, verbose=0)
        
        task_scheduler = torch.optim.lr_scheduler.ExponentialLR(task_optimizer, gamma=0.999)
        worker_scheduler = torch.optim.lr_scheduler.ExponentialLR(worker_optimizer, gamma=0.999)

        for epoch in range(epochs):

            for batch in train_loader:
                
                batch = [item.to(device) for item in batch]
                loss = self.training_step(batch)
                loss.backward()
                
                task_optimizer.step()
                task_optimizer.zero_grad()
                task_scheduler.step()

                worker_optimizer.step()
                worker_optimizer.zero_grad()
                worker_scheduler.step()

            if self.lambda1 > 0:
                self.P = task_kmeans.fit_predict(self.task_factors.weight)
            if self.lambda2 > 0:
                self.Q = worker_kmeans.fit_predict(self.worker_factors.weight)
            
            result = self._evaluate(train_loader, device)
            if print_history:
                self.epoch_end(epoch, result)
            history.append(result)
            
        return history


    
class fuse_MF(torch.nn.Module):

    """
    Matrix factorization for binary/continuous entries with Fused penalty
    """                

    def __init__(self, n_tasks, n_workers, n_factors=3, 
                    lambda11=1e-2, lambda12=1e-2, lambda21=1e-2, lambda22=1e-2, loss="BCE"):

        super().__init__()

        self.n_tasks = n_tasks
        self.n_workers = n_workers
        self.n_factors = n_factors
        self.loss_func = loss
        self.task_factors = torch.nn.Embedding(n_tasks, n_factors)
        self.worker_factors = torch.nn.Embedding(n_workers, n_factors)
        self.task_centroid = torch.nn.Embedding(n_tasks, n_factors) # centroids for each task
        self.worker_centroid = torch.nn.Embedding(n_workers, n_factors) # centroids for each worker
        self.lambda11 = lambda11
        self.lambda12 = lambda12
        self.lambda21 = lambda21
        self.lambda22 = lambda22

    def forward(self, task, worker):
    
        if self.loss_func == "BCE":
    
            score = torch.sigmoid((self.task_factors(task) * self.worker_factors(worker)).sum(1))

        elif self.loss_func == "MSE":

            score = (self.task_factors(task) * self.worker_factors(worker)).sum(1)

        return score

    def fuse_penalty(self, task_factors, worker_factors, task_centroids, worker_centroids):

        task_se = torch.sum(torch.square(task_factors - task_centroids))
        worker_se = torch.sum(torch.square(worker_factors - worker_centroids))

        task_fuse = 0

        for i in range(task_factors.size(0)):
            for j in range(i, task_factors.size(0)):

                task_fuse += torch.norm(task_centroids[i, ] - task_centroids[j, ], p=1)

        worker_fuse = 0
        for i in range(worker_factors.size(0)):
            for j in range(i, worker_factors.size(0)):

                worker_fuse += torch.norm(worker_centroids[i, ] - worker_centroids[j, ], p=1)

        total_penalty = self.lambda11 * task_se + \
                        self.lambda12 * task_fuse + \
                        self.lambda21 * worker_se + \
                        self.lambda22 * worker_fuse
        
        return total_penalty

    def training_step(self, batch):

        _, tasks, workers, labels = batch

        output = self(tasks, workers)

        if self.loss_func == "BCE":
            loss_func = torch.nn.BCELoss()
        elif self.loss_func == "MSE":
            loss_func = torch.nn.MSELoss()

        loss = loss_func(output, labels)

        penalty = self.fuse_penalty(self.task_factors(tasks),
                                    self.worker_factors(workers),
                                    self.task_centroid(tasks),
                                    self.worker_centroid(workers))

        return loss + penalty

    def _evaluate(self, train_loader, device):
            
        outputs = []
        for batch in train_loader:
            batch = [item.to(device) for item in batch]
            outputs.append(self.training_step(batch))
        
        return torch.stack(outputs).mean()

    def epoch_end(self, epoch, result):
    
        print("Epoch: {} - Training loss: {:.4f}".format(epoch, result))

    def fit(self, epochs, learning_rate, train_loader, print_history, opt_func,
                weight_decay, device):

        history = []
        task_f_optimizer = opt_func(self.task_factors.parameters(), learning_rate, weight_decay=weight_decay)
        worker_f_optimizer = opt_func(self.worker_factors.parameters(), learning_rate, weight_decay=weight_decay)

        task_c_optimizer = opt_func(self.task_centroid.parameters(), learning_rate, weight_decay=weight_decay)
        worker_c_optimizer = opt_func(self.worker_centroid.parameters(), learning_rate, weight_decay=weight_decay)

        task_f_optimizer.zero_grad()
        worker_f_optimizer.zero_grad()
        task_c_optimizer.zero_grad()
        worker_c_optimizer.zero_grad()

        for epoch in range(epochs):

            for batch in train_loader:

                batch = [item.to(device) for item in batch]
                loss = self.training_step(batch)
                loss.backward()

                task_c_optimizer.step()
                task_c_optimizer.zero_grad()

                worker_c_optimizer.step()
                worker_c_optimizer.zero_grad()
                
                task_f_optimizer.step()
                task_f_optimizer.zero_grad()

                worker_f_optimizer.step()
                worker_f_optimizer.zero_grad()

            result = self._evaluate(train_loader, device)
            if print_history:
                self.epoch_end(epoch, result)
            history.append(result)

        return history


        



                
        


