from abc import ABC, abstractmethod
from typing import Any

class QueryHandler(ABC):
    @abstractmethod
    def search(self, queries: list[str]) -> list[Any]:
        pass

class ConceptIDQueryHandler(QueryHandler):
    @abstractmethod
    def search(self, queries: list[str]) -> list[list[int]]:
        pass
