# -------------------------------------------------------------------
# Copyright (c) 2025 Year Anonymous Authors of NeurIPS 2025 Submission 10239
# -------------------------------------------------------------------
from abc import ABC, abstractmethod


class IROptimizer(ABC):
    def __init__(self) -> None:
        pass

    def step(self, *args, **kwargs) -> float:
        return NotImplemented

    @abstractmethod
    def cal_loss(self, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def cal_loss_graph(self,*args,**kwargs):
        return NotImplemented
    
    @abstractmethod
    def save(self, path):
        return NotImplemented

