import torch
from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self):
        self.name = self.__class__.__name__.lower()
        self.value = 0.0

    @abstractmethod
    def update(self, output, target):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class Accuracy(Metric):
    def __init__(self):
        self.corrects = 0
        self.num_instances = 0
        super().__init__()

    def reset(self):
        self.corrects = 0
        self.num_instances = 0

    def update(self, output, target):
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            self.num_instances += len(target)
            self.corrects += torch.sum(pred == target).item()

    def eval(self):
        if self.num_instances == 0:
            self.value = 0.0
        else:
            self.value = self.corrects / self.num_instances
        self.reset()


class TopKAccuracy(Metric):
    def __init__(self, k=3):
        self.corrects = 0
        self.k = k
        self.num_instances = 0
        self.value = 0.0
        super().__init__()

    def reset(self):
        self.corrects = 0
        self.num_instances = 0

    def update(self, output, target):
        with torch.no_grad():
            pred = torch.topk(output, self.k, dim=1)[1]
            assert pred.shape[0] == len(target)
            self.num_instances += len(target)
            for i in range(self.k):
                self.corrects += torch.sum(pred[:, i] == target).item()

    def eval(self):
        if self.num_instances == 0:
            self.value = 0.0
        else:
            self.value = self.corrects / self.num_instances
        self.reset()
