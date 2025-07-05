from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RetrievalQuery:
    query_text: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10


@dataclass
class RetrievalResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    source_type: str


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve relevant information based on query"""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize retriever with configuration"""
        pass