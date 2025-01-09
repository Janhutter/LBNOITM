from abc import ABC, abstractmethod

class Reducer(ABC):
    def __init__(self, model_name=None):
        self.model_name = model_name

    @abstractmethod
    def __call__(self, kwargs):
        pass

    @abstractmethod
    def reduce_fn(self, doc_embs, docs):
         pass