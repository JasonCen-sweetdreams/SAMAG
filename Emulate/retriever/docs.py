from loguru import logger
from Emulate.retriever import retriever_registry
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.retrievers import BaseRetriever
from typing import (
    List,
    Dict,
    Callable,
    Optional,
    Union,
    Any)
from langchain_core.pydantic_v1 import root_validator, Field
from langchain_core.documents import Document
from functools import cmp_to_key
import torch
from torch_geometric.data import Data, HeteroData
from collections import defaultdict, deque, Counter
import networkx as nx


def compare_article_items(item1, item2):
    try:
        if abs(item1[1]-item2[1]>0.05):
            return item2[1] - item1[1]
        else:
            return item2[0].metadata["cited"] - item1[0].metadata["cited"]
    except:
        return item2[1] - item1[1]
    

def compare_article_items(item1, item2):
    try:
        if abs(item1[1]-item2[1]>0.05):
            return item2[1] - item1[1]
        else:
            return 1 if item2[0].metadata["time"] > item1[0].metadata["time"] else -1
    except:
        return item2[1] - item1[1]
    
def compare_movie_items(item1, item2):
    try:
        if abs(item1[1]-item2[1]>0.05):
            return item2[1] - item1[1]
        else:
            return item2[0].metadata["Timestamp"] - item1[0].metadata["Timestamp"]
    except:
        return item2[1] - item1[1]  

def compare_social_items(item1, item2):
    try:
        return item2[1] - item1[1]
    except:
        raise NotImplementedError("Only support score_cite==True for social environment!")
        



@retriever_registry.register("graph_vector_retriever")
class GraphVectorRetriever(VectorStoreRetriever):
    
    compare_function: Callable = None
    cache: dict = {}
    
    
    def __init__(self,
                 compare_function_type: str, # article / movie
                 **kwargs):
        compare_function_map ={
            "article": compare_article_items,
            "movie": compare_movie_items,
            "social": compare_social_items
        }
        compare_function = compare_function_map.get(compare_function_type)

        
        super().__init__(compare_function = compare_function,
                         **kwargs)
    

        
        
    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        search_type = values["search_type"]
        if search_type not in cls.allowed_search_types:
            raise ValueError(
                f"search_type of {search_type} not allowed. Valid values are: "
                f"{cls.allowed_search_types}"
            )
        if search_type == "similarity_score_threshold":
            score_threshold = values["search_kwargs"].get("score_threshold")
            if (score_threshold is None) or (not isinstance(score_threshold, float)):
                raise ValueError(
                    "`score_threshold` is not specified with a float value(0~1) "
                    "in `search_kwargs`."
                )
        
        if values["search_kwargs"].get("score_cite") is not None:
            score_cite =  values["search_kwargs"].get("score_cite")
            if score_cite and search_type == "mmr":
                raise NotImplementedError("'score_cite == True' is not supported for mmr searching")
        
        return values
    
    def _get_relevant_documents(
        self, query: str, *, run_manager
    ) :
        
        if "similarity" in self.search_type:
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            score_cite = self.search_kwargs.get("score_cite",False)
            if score_cite:
                docs_and_similarities = sorted(docs_and_similarities, 
                key=cmp_to_key(self.compare_function))
            docs = [doc for doc,_ in docs_and_similarities]
            
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs


    @classmethod
    def from_db(cls, 
                vectorstore,
                 **kwargs):
        tags = kwargs.pop("tags", None) or []
        tags.extend(vectorstore._get_retriever_tags())
        return cls(vectorstore = vectorstore,
                   **kwargs,
                   tags = tags
                   )
        
@retriever_registry.register("hybrid_graph_vector_retriever_for_article")
class HybridGraphVectorRetriever(VectorStoreRetriever):
    """Hybrid retriever: FAISS semantic + Graph neighbor expansion."""
    graph_retriever: Any = Field(default=None, description="Graph-based neighbor retriever")
    compare_function: Optional[Callable] = None
    graph_hops: int = 1  # Default 1-hop
    # Expansion strategy
    expansion_strategy: str = "simple"   # "community"

    def __init__(self,
                 compare_function_type: str,
                 graph_retriever: Any,
                 graph_hops: int = 1,
                 expansion_strategy: str = "simple",
                 **kwargs):
        compare_function_map = {
            "article": compare_article_items,
            "movie": compare_movie_items,
            "social": compare_social_items
        }
        compare_function = compare_function_map.get(compare_function_type)
        super().__init__(
            compare_function=compare_function,
            graph_retriever=graph_retriever,
            graph_hops=graph_hops,
            expansion_strategy=expansion_strategy,
            **kwargs
        )
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List:
        # 1) semantic retrieval
        if "similarity" in self.search_type:
            docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
            if self.search_kwargs.get("score_cite", False):
                docs_and_scores = sorted(docs_and_scores, key=cmp_to_key(self.compare_function))
            initial_docs = [doc for doc, _ in docs_and_scores]
        elif self.search_type == "mmr":
            initial_docs = self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)
        else:
            raise ValueError(f"search_type {self.search_type} not supported.")

        # 2) Graph-neighborhood expansion
        expanded = []
        # Community detection
        if self.expansion_strategy == "community":
            if hasattr(self.graph_retriever, 'retrieve_by_common_neighbors'):
                expanded = self.graph_retriever.retrieve_by_common_neighbors(initial_docs, hops=self.graph_hops)
            else:
                logger.warning("Community expansion strategy selected, but retriever does not have 'retrieve_by_common_neighbors' method. Falling back to simple expansion.")
                for doc in initial_docs:
                    title = doc.metadata.get("title")
                    if title:
                        expanded.extend(self.graph_retriever.retrieve_by_title(title, hops=self.graph_hops))

        
        elif self.expansion_strategy == "simple":
            for doc in initial_docs:
                title = doc.metadata.get("title")
                if title:
                    expanded.extend(self.graph_retriever.retrieve_by_title(title, hops=self.graph_hops))
        
        else:
            raise ValueError(f"Unknown expansion_strategy: {self.expansion_strategy}")

        # 3) Deduplicate and merge
        seen, final = set(), []
        for d in initial_docs + expanded:
            t = d.metadata.get("title")
            if t and t not in seen:
                seen.add(t)
                final.append(d)
        return final

    @classmethod
    def from_db(cls, vectorstore, graph_retriever, graph_hops=1, expansion_strategy="simple", **kwargs):
        tags = kwargs.pop("tags", None) or []
        tags.extend(vectorstore._get_retriever_tags())
        return cls(
            vectorstore=vectorstore,
            graph_retriever=graph_retriever,
            graph_hops=graph_hops,
            expansion_strategy=expansion_strategy,
            **kwargs,
            tags=tags,
        )
    

@retriever_registry.register("graph_structure_retriever_for_article")
class GraphRAGRetriever:
    """
    Graph-based literature retriever over a heterogeneous graph with two node types: 'article' and 'author'.
    Provides explainable retrieval path information.
    """
    def __init__(
        self,
        pyg_graph_data: HeteroData,
        node_id_to_title: Dict[int, str],
        title_to_node_id: Dict[str, int],
        index_to_doc: Dict[int, Document],
        edge_types: Optional[List[str]] = None,
        allow_co_citation: bool = False,
        allow_co_author: bool = False,
    ):
        self.graph = pyg_graph_data
        self.node_id_to_title = node_id_to_title
        self.title_to_node_id = title_to_node_id
        self.index_to_doc = index_to_doc
        self.edge_types = edge_types or ["cites", "authored_by", "writes"]
        self.allow_co_citation = allow_co_citation
        self.allow_co_author = allow_co_author

    def retrieve_by_title(self, title: str, hops: int = 1) -> List[Document]:
        """
        Starting from the given title, return multi-hop neighbor articles and
        attach the retrieval paths to each document's metadata.
        """
        start_idx = self.title_to_node_id.get(title)
        if start_idx is None:
            return []

        retrieved: Dict[int, Dict[str, Any]] = {}

        current = {start_idx}
        for hop in range(hops):
            next_level = set()
            # article -> cites -> article
            if "cites" in self.edge_types:
                src, dst = self.graph['article', 'cites', 'article'].edge_index
                for u, v in zip(src.tolist(), dst.tolist()):
                    if u in current and v not in retrieved:
                        reason = f"hop_{hop+1}: {self.node_id_to_title[u]} cites {self.node_id_to_title[v]}"
                        next_level.add(v)
                        retrieved.setdefault(v, {"paths": []})["paths"].append(reason)
            # article <- cites <- article
            if self.allow_co_citation:
                src, dst = self.graph['article', 'cites', 'article'].edge_index
                for u, v in zip(src.tolist(), dst.tolist()):
                    if v in current and u not in retrieved:
                        reason = f"hop_{hop+1}: {self.node_id_to_title[u]} cited by {self.node_id_to_title[v]}"
                        next_level.add(u)
                        retrieved.setdefault(u, {"paths": []})["paths"].append(reason)

            if self.allow_co_author and "writes" in self.graph.node_types:
                # article <- writes - author
                aw_src, aw_dst = self.graph['author', 'writes', 'article'].edge_index
                # article -> authored_by -> author
                for aid, art in zip(aw_src.tolist(), aw_dst.tolist()):
                    if art in current:
                        # author aid 写过 article art
                        for a2, art2 in zip(aw_src.tolist(), aw_dst.tolist()):
                            if a2 == aid and art2 not in retrieved and art2 not in current:
                                reason = f"hop_{hop+1}: co-author via author_{aid} wrote {self.node_id_to_title[art2]}"
                                next_level.add(art2)
                                retrieved.setdefault(art2, {"paths": []})["paths"].append(reason)

            current = next_level

        docs: List[Document] = []
        for idx, info in retrieved.items():
            doc = self.index_to_doc.get(idx)
            if doc:
                doc.metadata = dict(doc.metadata)
                doc.metadata['retrieval_paths'] = info['paths']
                docs.append(doc)
        return docs

    @classmethod
    def from_db(
        cls,
        pyg_graph_data: HeteroData,
        node_id_to_title: Dict[int, str],
        title_to_node_id: Dict[str, int],
        index_to_doc: Dict[int, Document],
        edge_types: Optional[List[str]] = None,
        allow_co_citation: bool = False,
        allow_co_author: bool = False,
        **kwargs
    ):
        return cls(
            pyg_graph_data=pyg_graph_data,
            node_id_to_title=node_id_to_title,
            title_to_node_id=title_to_node_id,
            index_to_doc=index_to_doc,
            edge_types=edge_types,
            allow_co_citation=allow_co_citation,
            allow_co_author=allow_co_author,
        )


@retriever_registry.register("community_graph_retriever_for_article")
class CommunityGraphRetriever(GraphRAGRetriever):
    """
    Based on common neighbors.
    Goal: find community core nodes that connect multiple anchors.
    Reuses GraphRAGRetriever's data initialization.
    """
    def __init__(self, pyg_graph_data, node_id_to_title, title_to_node_id, index_to_doc, edge_types = None, allow_co_citation = False, allow_co_author = True):
        super().__init__(pyg_graph_data, node_id_to_title, title_to_node_id, index_to_doc, edge_types, allow_co_citation, allow_co_author)

    def retrieve_by_common_neighbors(self, anchor_docs: List, hops: int = 1) -> List:
        if not anchor_docs:
            print(f"Anchor docs are needed!")
            return

        anchor_titles = {doc.metadata.get("title") for doc in anchor_docs if doc.metadata.get("title")}
        anchor_indices = {self.title_to_node_id.get(title) for title in anchor_titles if self.title_to_node_id.get(title) is not None}

        if not anchor_indices:
            print(f"anchor indices error! anchor_titles: {anchor_titles}")
            return

        all_neighbors_docs = []
        for start_idx in anchor_indices:
            one_hop_docs = self.retrieve_by_title(self.node_id_to_title[start_idx], hops=hops)
            all_neighbors_docs.extend(one_hop_docs)
        
        neighbor_titles = [doc.metadata.get("title") for doc in all_neighbors_docs if doc.metadata.get("title")]
        common_neighbor_count = Counter(neighbor_titles)

        for title in anchor_titles:
            if title in common_neighbor_count:
                del common_neighbor_count[title]
        
        retrieved_docs_map = {doc.metadata.get("title"): doc for doc in all_neighbors_docs}

        final_docs = []
        seen_titles = set()

        for title, count in common_neighbor_count.most_common():
            if title in retrieved_docs_map and title not in seen_titles:
                doc = retrieved_docs_map[title]
                doc.metadata['common_neighbor_count'] = count
                final_docs.append(doc)
                seen_titles.add(title)
        
        return final_docs

    @classmethod
    def from_db(cls, **kwargs: Any):
        return cls(**kwargs)


@retriever_registry.register("graph_structure_retriever_for_social")
class GraphSocialRetriever:
    def __init__(
        self,
        data: HeteroData,
        idx_to_doc: Dict[int, "Document"],
        allow_retweet: bool = True
    ):
        """
        data: HeteroData returned by build_social_graph
        idx_to_doc: mapping tweet_idx -> Document
        max_hops: default one-hop; can be set to multi-hop
        """
        self.data = data
        self.idx_to_doc = idx_to_doc
        self.allow_retweet = allow_retweet

    def retrieve_by_user(self,
                         user_id: int,
                         depth: int = 1,
                         top_k: int = 10) -> List[Document]:
        """
        1) BFS starting from user_id up to 'depth' layers, collecting all neighbor users within that depth
        2) For each layer's neighbor users, collect their 'tweets' and (optionally) 'retweets'
        3) Exclude tweets whose original author (metadata['user_index']) equals the caller user_id
        4) Sort candidates as needed and return top_k (here: simple truncation without scoring)
        """
        visited_users = set([user_id])
        current_frontier = {user_id}
        all_neighbor_users = set()

        for _ in range(depth):
            next_frontier = set()
            # (user->follow->user)
            if ('user','follow','user') in self.graph.edge_types:
                src_f, dst_f = self.graph['user','follow','user'].edge_index
                for u,v in zip(src_f.tolist(), dst_f.tolist()):
                    if u in current_frontier and v not in visited_users:
                        next_frontier.add(v)

            # (user->friend->user)
            if ('user','friend','user') in self.graph.edge_types:
                src_fr, dst_fr = self.graph['user','friend','user'].edge_index
                for u,v in zip(src_fr.tolist(), dst_fr.tolist()):
                    if u in current_frontier and v not in visited_users:
                        next_frontier.add(v)

            visited_users |= next_frontier
            all_neighbor_users |= next_frontier
            current_frontier = next_frontier
            if not current_frontier:
                break


        candidate_tweet_ids = set()

        # user->tweets->tweet
        if ('user','tweets','tweet') in self.graph.edge_types:
            src_ut, dst_ut = self.graph['user','tweets','tweet'].edge_index
            for u, t in zip(src_ut.tolist(), dst_ut.tolist()):
                if u in all_neighbor_users:
                    candidate_tweet_ids.add(t)

        # user->retweets->tweet
        if self.allow_retweet and ('user','retweets','tweet') in self.graph.edge_types:
            src_urt, dst_urt = self.graph['user','retweets','tweet'].edge_index
            for u, t in zip(src_urt.tolist(), dst_urt.tolist()):
                if u in all_neighbor_users:
                    candidate_tweet_ids.add(t)

        filtered_tweet_ids = []
        for t in candidate_tweet_ids:
            doc = self.idx_to_doc.get(t)
            if doc is None:
                continue
            original_author = int(doc.metadata.get("user_index", -1))
            if original_author == user_id:
                continue
            filtered_tweet_ids.append(t)

        top_ids = filtered_tweet_ids[:top_k]
        results = [self.idx_to_doc[t] for t in top_ids if t in self.idx_to_doc]
        return results
    
    @classmethod
    def from_db(
        cls,
        data: HeteroData,
        idx_to_doc: Dict[int, Document],
        allow_retweet: bool = True,
        **kwargs
    ):
        return cls(
            data=data,
            idx_to_doc=idx_to_doc,
            allow_retweet=allow_retweet
        )

@retriever_registry.register("hybrid_graph_vector_retriever_for_social")
class HybridSocialRetriever(BaseRetriever):
    """
    Hybrid social retriever.

    Maintains:
      - vector_retriever: FAISS vector retriever instance
      - graph_retriever: social-graph retriever instance

    Supports two independent retrieval modes:
      1) Vector retrieval (query)
      2) Graph-structure retrieval (user_id)
    """
    vector_retriever: Any
    graph_retriever: Any
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)  # 默认空 dict
    search_type: str = "similarity"

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vector_retriever: Any,
        graph_retriever: Any,
        **kwargs
    ):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.graph_retriever  = graph_retriever

        self.search_kwargs = getattr(vector_retriever, "search_kwargs", {})
        self.search_type   = getattr(vector_retriever, "search_type", "similarity")

    def _get_relevant_documents(
        self,
        query: str = None,
        user_id: int = None,
        k_vector: int = 10,
        k_graph: int = 10,
        depth: int = 1,
        run_manager=None
    ) -> List[Document]:
        """
        Hybrid retrieval:
        - If `query` is provided, perform vector retrieval and take top k_vector.
        - If `user_id` is provided, perform graph retrieval and take top k_graph.
        Return the merged result list.
        """
        results: List[Document] = []

        if query and self.vector_retriever:
            docs = self.vector_retriever.get_relevant_documents(query)
            results.extend(docs[:k_vector])

        if user_id is not None and self.graph_retriever:
            docs = self.graph_retriever.retrieve_by_user(
                user_id=user_id, depth=depth, top_k=k_graph
            )
            results.extend(docs)
        return results

    @classmethod
    def from_db(
        cls,
        vector_retriever: Any,
        graph_retriever: Any,
        **kwargs
    ):
        return cls(
            vector_retriever=vector_retriever,
            graph_retriever=graph_retriever,
            **kwargs
        )


@retriever_registry.register("movie_graph_rag_retriever")
class MovieGraphRAGRetriever:
    """
    Graph-only retriever over the movie heterogeneous graph.
    Mainly used for graph-structure-based personalized recommendation (e.g., collaborative filtering).
    """
    def __init__(
        self,
        pyg_graph_data: HeteroData,
        graph_maps: Dict,
        docs_map: Dict[int, Document],
    ):
        self.graph_data = pyg_graph_data
        self.graph_maps = graph_maps
        self.docs_map = docs_map
        self._nx_graph: Optional[nx.Graph] = None

    def _get_nx_graph(self) -> Optional[nx.Graph]:
        if self._nx_graph is None and self.graph_data:
            try:
                from torch_geometric.utils import to_networkx
                logger.info("Converting HeteroData to NetworkX for retrieval...")
                graph_subset = self.graph_data['user', 'rated', 'movie']
                self._nx_graph = to_networkx(graph_subset, edge_attrs=['edge_attr'])

                rev_user_map = {v: k for k, v in self.graph_maps['user'].items()}
                rev_movie_map = {v: k for k, v in self.graph_maps['movie'].items()}
                
                node_mapping = {
                    **{i: f"u_{rev_user_map.get(i)}" for i in range(len(rev_user_map))},
                    **{i + len(rev_user_map): f"m_{rev_movie_map.get(i)}" for i in range(len(rev_movie_map))}
                }
                
                nx.relabel_nodes(self._nx_graph, node_mapping, copy=False)
                logger.info("Conversion finished.")
            except Exception as e:
                logger.error(f"Failed to convert HeteroData to NetworkX: {e}")
                return None
        return self._nx_graph

    def retrieve_by_user_id(self, user_id: int, k: int = 5) -> List[Document]:
        """
        Personalized recommendation via collaborative filtering.
        """
        graph = self._get_nx_graph()
        if not graph or not user_id:
            return []

        user_node = f"u_{user_id}"
        if user_node not in graph:
            return []

        try:
            seen_movies = {n for n in graph.neighbors(user_node)}
            recommendations = defaultdict(int)

            for movie_node in seen_movies:
                if graph.edges[user_node, movie_node].get('edge_attr', [0])[0] >= 4.0:
                    for similar_user_node in graph.neighbors(movie_node):
                        if similar_user_node == user_node: continue
                        for rec_movie_node in graph.neighbors(similar_user_node):
                            if rec_movie_node not in seen_movies and rec_movie_node.startswith('m_'):
                                if graph.edges[similar_user_node, rec_movie_node].get('edge_attr', [0])[0] >= 4.0:
                                    recommendations[rec_movie_node] += 1

            sorted_recs = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)
            
            results = []
            for movie_node, score in sorted_recs[:k]:
                movie_id = int(movie_node.split('_')[1])
                if movie_id in self.docs_map:
                    results.append(self.docs_map[movie_id])
            return results
        except Exception as e:
            logger.error(f"Collaborative filtering failed for user {user_id}: {e}")
            return []

    @classmethod
    def from_db(cls, **kwargs):
        return cls(**kwargs)

@retriever_registry.register("hybrid_movie_retriever")
class HybridMovieRetriever(BaseRetriever):
    vector_retriever: BaseRetriever
    graph_retriever: MovieGraphRAGRetriever
    k: int = 10
    k_vector: int = 10
    k_graph: int = 10 
    search_kwargs: Dict = {}

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any, **kwargs: Any
    ) -> List[Document]:
        
        user_id = kwargs.get("user_id")
        
        # 1) Vector-retrieval path (content-based)
        vector_results = self.vector_retriever.get_relevant_documents(query, **kwargs)
        
        graph_results = []
        2) Graph-retrieval path
        if user_id:
            graph_results = self.graph_retriever.retrieve_by_user_id(user_id=user_id, k=self.k_graph)

        # 3) Merge and deduplicate
        final_results = []
        seen_ids = set()

        for doc in graph_results[:self.k_graph]:
            movie_id = doc.metadata.get("MovieId")
            if movie_id not in seen_ids:
                final_results.append(doc)
                seen_ids.add(movie_id)

        for doc in vector_results[:self.k_vector]:
            movie_id = doc.metadata.get("MovieId")
            if movie_id not in seen_ids:
                final_results.append(doc)
                seen_ids.add(movie_id)

        return final_results
    
    @classmethod
    def from_db(cls, **kwargs):
        return cls(**kwargs)