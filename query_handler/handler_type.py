from abc import ABC, abstractmethod
from typing import List

class QueryHandler(ABC):
    @abstractmethod
    def search(self, queries: List[str]) -> List:
        pass

class ConceptIDQueryHandler(QueryHandler):
    @abstractmethod
    def search(self, queries: List[str]) -> List[List[int]]:
        pass
