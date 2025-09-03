from LLMGraph.registry import Registry
retriever_registry = Registry(name="RetrieverRegistry")

from .docs import GraphVectorRetriever, HybridGraphVectorRetriever, GraphRAGRetriever
from .arxiv import GraphArxivRetriever
from .google_scholar import GraphGoogleScholarRetriever