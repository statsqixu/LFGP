from torch.utils.data import Dataset

class LFGPDataset(Dataset):

    def __init__(self, task, worker, label):

        self.task, self.worker, self.label = task, worker, label

    def __len__(self):

        return len(self.task)

    def __getitem__(self, idx):

        return idx, self.task[idx], self.worker[idx], self.label[idx]

