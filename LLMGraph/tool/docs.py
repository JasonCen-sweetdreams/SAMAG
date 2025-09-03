from functools import partial
from typing import Any, Optional, Type, List


from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.retrievers import BaseRetriever
from loguru import logger
from .tool_warpper import GraphServiceFactory
from . import RetrieverInput
from agentscope.service import ServiceResponse, ServiceExecStatus
from langchain_core.documents import Document
import functools
import random

class ArticleInfoInput(BaseModel):
    """Input to the retriever."""

    title: str = Field(description="title of your paper")
    keyword: str = Field(description="the keywords of your paper")
    abstract: str = Field(description="the abstract of your paper")
    citation: str = Field(description="the citations")


"""sort 精排"""
def generate_compare_function_big_name(big_name_list):
    # 在interested topic中的排前面
    def compare(item1, item2):
        # 根据interested_topic的长度来决定项目的排序权重
        item1_weight = 1 if item1.metadata.get("author_name") in big_name_list else 0
        item2_weight = 1 if item2.metadata.get("author_name") in big_name_list else 0
        return item2_weight - item1_weight
    return compare

def generate_compare_function_topic(interested_topic):
    def compare(item1, item2):
        # 根据interested_topic的长度来决定项目的排序权重
        # logger.info(f"item1 metadata: {item1.metadata}, \nitem2 metadata: {item2.metadata}")
        topics = interested_topic or []
        item1_weight = 1 if item1.metadata.get("topic") in topics else 0
        item2_weight = 1 if item2.metadata.get("topic") in topics else 0
        return item2_weight - item1_weight
    return compare

def generate_compare_function_cite():
    def compare(item1, item2):
        # 根据interested_topic的长度来决定项目的排序权重
        try:
            return item2[0].metadata["cited"] - item1[0].metadata["cited"]
        except:
            return -1

    return compare

def generate_compare_function_semantic(query, embedding_model, docs: List[Document]):
    if hasattr(embedding_model, "embed_query"):
        query_emb = embedding_model.embed_query(query)
    else:
        query_emb = embedding_model.embed_documents([query])[0]
    
    contents = [doc.page_content for doc in docs]
    docs_embs = embedding_model.embed_documents(contents)
    sims = cosine_similarity([query_emb], docs_embs)[0]
    sim_map = {id(doc): sim for doc, sim in zip(docs, sims)}

    def compare(item1, item2):
        sim1 = sim_map.get(id(item1), 0.0)
        sim2 = sim_map.get(id(item2), 0.0)

        return sim2 - sim1

    return compare

def generate_compare_function_community():
    """
    生成一个基于社区紧密度的比较函数。
    优先排序那些作为更多锚点论文共同邻居的文档。
    """
    def compare(item1, item2):
        # common_neighbor_count 是我们在 CommunityGraphRetriever 中添加的元数据
        # 这个值越高，说明该文档与初始语义锚点集的结构关联越紧密
        score1 = item1.metadata.get("common_neighbor_count", 0)
        score2 = item2.metadata.get("common_neighbor_count", 0)
        return score2 - score1
    return compare

def generate_compare_function_hybrid(
        query: str,
        embedding_model: Any,
        docs: List[Document],
        alpha: float = 0.5
) -> callable:
    """
    生成一个混合了社区分数和语义相似度分数的比较函数。

    Args:
        query (str): 用于计算语义相似度的用户查询。
        embedding_model (Any): 嵌入模型。
        docs (List[Document]): 待排序的文档列表。
        alpha (float): 语义分数的权重，取值范围 [0, 1]。
                       社区分数的权重将是 (1 - alpha)。
                       alpha = 1.0 -> 只看语义
                       alpha = 0.0 -> 只看社区
                       alpha = 0.5 -> 两者同等重要

    Returns:
        callable: 一个可用于 sorted() 的比较函数。
    """
    if hasattr(embedding_model, "embed_query"):
        query_emb = embedding_model.embed_query(query)
    else:
        query_emb = embedding_model.embed_documents([query])[0]

    contents = [doc.page_content for doc in docs]
    docs_embs = embedding_model.embed_documents(contents)
    semantic_scores_raw = cosine_similarity([query_emb], docs_embs)[0]

    community_scores_raw = [doc.metadata.get("common_neighbor_count", 0) for doc in docs]

    min_sem, max_sem = min(semantic_scores_raw), max(semantic_scores_raw)
    normalized_sem_scores = []
    if max_sem > min_sem:
        normalized_sem_scores = [(s - min_sem) / (max_sem - min_sem) for s in semantic_scores_raw]
    else:
        normalized_sem_scores = [0.5] * len(docs)
    
    min_com, max_com = min(community_scores_raw), max(community_scores_raw)
    normalized_com_scores = []
    if max_com > min_com:
        normalized_com_scores = [(c - min_com) / (max_com - min_com) for c in community_scores_raw]
    else:
        normalized_com_scores = [0.5] * len(docs)
    
    hybrid_scores = {}
    for i, doc in enumerate(docs):
        final_score = (alpha * normalized_sem_scores[i]) + \
                      ((1 - alpha) * normalized_com_scores[i])
        hybrid_scores[id(doc)] = final_score
    
    def compare(item1, item2):
        score1 = hybrid_scores.get(id(item1), -1)
        score2 = hybrid_scores.get(id(item2), -1)
        if score2 > score1:
            return 1
        elif score1 > score2:
            return -1
        else:
            return 0
    
    return compare

def format_document(doc: Document, 
                    article_meta_data:dict,
                    author_data:dict,
                    prompt: BasePromptTemplate[str],
                    experiment:list = [], # default/shuffle/false cite
                    ) -> str:
    """Format a document into a string based on a prompt template.

    First, this pulls information from the document from two sources:

    1. `page_content`:
        This takes the information from the `document.page_content`
        and assigns it to a variable named `page_content`.
    2. metadata:
        This takes information from `document.metadata` and assigns
        it to variables of the same name.

    Those variables are then passed into the `prompt` to produce a formatted string.

    Args:
        doc: Document, the page_content and metadata will be used to create
            the final string.
        prompt: BasePromptTemplate, will be used to format the page_content
            and metadata into the final string.

    Returns:
        string of the document formatted.

    Example:
        .. code-block:: python

            from langchain_core.documents import Document
            from langchain_core.prompts import PromptTemplate

            doc = Document(page_content="This is a joke", metadata={"page": "1"})
            prompt = PromptTemplate.from_template("Page {page}: {page_content}")
            format_document(doc, prompt)
            >>> "Page 1: This is a joke"
    """
    choose_key = "cited"
    
    try:
        title = doc.metadata["title"]
        author_ids = article_meta_data[title]["author_ids"]
        author_ids = list(filter(lambda x:x in author_data.keys(), author_ids))

        best_author_idx = author_ids[0]
        for author_id in author_ids:
            author_info = author_data[author_id]
            if author_data[best_author_idx].get(choose_key,0) < author_info.get(choose_key,0):
                best_author_idx = author_id
        
        best_author_info ={
            "author_cited": author_data[best_author_idx].get("cited",0),
            "country": author_data[best_author_idx].get("country",0),
            "institution": author_data[best_author_idx].get("institution",0),
            "author_name": author_data[best_author_idx].get("name",0),
        }
    except:
        best_author_info ={
                    "author_cited": "Unknown",
                    "country":"",
                    "institution": "",
                    "author_name": "Unknown",
                }
    base_info ={
        **doc.metadata,
        "page_content":doc.page_content,
        **best_author_info
    }

    missing_metadata = set(prompt.input_variables).difference(base_info)
    if len(missing_metadata) > 0:
        required_metadata = [
            iv for iv in prompt.input_variables if iv != "page_content"
        ]
        raise ValueError(
            f"Document prompt requires documents to have metadata variables: "
            f"{required_metadata}. Received document with missing metadata: "
            f"{list(missing_metadata)}."
        )
    document_info = {k: base_info[k] for k in prompt.input_variables}
    if "false_data" in experiment:
        document_info["cited"] = random.randint(0, 2000)
    if "no_cite" in experiment:
        document_info["cited"] = "Unknown"
    if "no_content" in experiment:
        document_info["page_content"] = "Unknown"
    if "no_paper_time" in experiment:
        document_info["time"] = "Unknown"
    if "no_author" in experiment:
        document_info["author_name"] = "Unknown"
        document_info["author_cited"] = "Unknown"
    if "no_country" in experiment:
        document_info["country"] = "Unknown"
        document_info["institution"] = "Unknown"
    if "no_topic" in experiment:
        document_info["topic"] = "Unknown"
    if "anonymous" in experiment:
        document_info["author_name"] = "Unknown"
        document_info["country"] = "Unknown"
        document_info["institution"] = "Unknown"
        document_info["author_cited"] = "Unknown"
    
    return prompt.format(**document_info)



# def _get_article_relevant_documents(
#     query: str,
#     retriever: BaseRetriever,
#     article_meta_data:dict,
#     author_data:dict,
#     document_prompt: BasePromptTemplate,
#     document_separator: str,
#     experiment:list = [], # default/shuffle/false cite
#     filter_keys: list = [
#         "topic", "big_name", "write_topic"
#     ],
#     max_search:int = 5,
#     big_name_list:list = [],
#     interested_topics:List[str] = [],
#     research_topic:str =""
# ) -> str:
#     """Search for relevant papers, so as to refine your paper. \
# These papers should be included in your paper's citations if you use them in your paper. 

#     Args:
#         query (str): keywords split by commas. The informations about the papers you want to cite, you can enter some keywords or other info.

#     Returns:
#         str: information about some searched papers.
#     """
#     try:
#         k = retriever.search_kwargs["k"]
#         filtered_docs = []
#         query_list = query.split(",")
#         query_list.append(research_topic)
#         for query in query_list[:max_search]:
#             # docs = retriever.get_relevant_documents(query)
#             docs = retriever.get_relevant_documents(query)
#             filtered_docs.extend(docs)
        
#         filter_pipeline = []
#         filter_keys_set_map = {
#             "big_name":generate_compare_function_big_name(big_name_list), 
#             "topic":generate_compare_function_topic(interested_topics),
#             "write_topic":generate_compare_function_topic([research_topic]),
#             # "cite":generate_compare_function_cite(),
#             }
#         for filter_key,filter_function in filter_keys_set_map.items():
#             if filter_key in filter_keys:
#                 filter_pipeline.append(
#                     filter_keys_set_map[filter_key]
#                 )
#         if "nofilter" in experiment:
#             filter_pipeline = []
#         for filter_function in filter_pipeline:
#             key_func = functools.cmp_to_key(filter_function)
#             filtered_docs = list(
#             sorted(filtered_docs,
#                     key=key_func))
            
#         if len(filtered_docs)> k:
#             filtered_docs = filtered_docs[:k]

#         if  "shuffle" in experiment:
#             random.shuffle(filtered_docs)
            
#         output = document_separator.join(
#             format_document(doc, article_meta_data, author_data, document_prompt,
#                             experiment = experiment) for doc in filtered_docs
#         )
#         return ServiceResponse(status=ServiceExecStatus.SUCCESS,
#                            content=output)
#     except Exception as e:
#         return ServiceResponse(status=ServiceExecStatus.ERROR,
#                            content=e)

def _get_article_relevant_documents(
    query: str,
    retriever: BaseRetriever,
    article_meta_data:dict,
    author_data:dict,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    experiment:list = [], # default/shuffle/false cite
    filter_keys: list = [
        "topic", "big_name", "write_topic"
    ],
    max_search:int = 5,
    big_name_list:list = [],
    interested_topics:List[str] = [],
    research_topic:str ="",
    embedding_model: Any = None,
    hybrid_alpha: float = 0.5
) -> str:
    """Search for relevant papers, so as to refine your paper. \
These papers should be included in your paper's citations if you use them in your paper. 

    Args:
        query (str): keywords split by commas. The informations about the papers you want to cite, you can enter some keywords or other info.

    Returns:
        str: information about some searched papers.
    """
    try:
        k = retriever.search_kwargs["k"]
        filtered_docs = []
        query_list = query.split(",")
        query_list.append(research_topic)
        for query in query_list[:max_search]:
            # docs = retriever.get_relevant_documents(query)
            docs = retriever.get_relevant_documents(query)
            filtered_docs.extend(docs)
        total_len = len(filtered_docs)
        filtered_docs = [doc for doc in filtered_docs if doc is not None and isinstance(doc.metadata, dict)]
        preprocessed_len = len(filtered_docs)

        # 去重
        seen_titles = set()
        unique_docs = []
        for doc in filtered_docs:
            title = doc.metadata.get("title")
            if title and title not in seen_titles:
                unique_docs.append(doc)
                seen_titles.add(title)
        filtered_docs = unique_docs
        unique_len = len(unique_docs)
        logger.info(f"After preprocession, searched docs before fine ranking: {total_len} // {preprocessed_len} /// {unique_len}")
        logger.info(f"interested_topics: {interested_topics}")

        try:
            filter_pipeline = []
            filter_keys_set_map = {
                "big_name":generate_compare_function_big_name(big_name_list), 
                "topic":generate_compare_function_topic(interested_topics),
                "write_topic":generate_compare_function_topic([research_topic]),
                "cite":generate_compare_function_cite(),
                "community": generate_compare_function_community(),
                }

            # if "semantic" in filter_keys:
            #     if embedding_model is not None:
            #         filter_keys_set_map["semantic"] = generate_compare_function_semantic(query, embedding_model, filtered_docs) # TBD
            for filter_key,filter_function in filter_keys_set_map.items():
                if filter_key in filter_keys:
                    filter_pipeline.append(
                        filter_keys_set_map[filter_key]
                    )

            for filter_function in filter_pipeline:
                logger.info(f"{filter_function} fine rank about to process!")
                key_func = functools.cmp_to_key(filter_function)
                filtered_docs = list(
                sorted(filtered_docs,
                        key=key_func))
                logger.info(f"{filter_function} fine rank done!")
            ### 最后进行语义重排序
            if "semantic" in filter_keys:
                if embedding_model is not None:
                    sem_cmp = generate_compare_function_semantic(query, embedding_model, filtered_docs)
                    filtered_docs = sorted(filtered_docs, key=functools.cmp_to_key(sem_cmp))
                logger.info(f"semantic fine rank done!")
            
            if "hybrid_sort_com_sem" in filter_keys and embedding_model is not None:
                logger.info(f"Hybrid (semantic & community) fine rank about to process with alpha={hybrid_alpha}!")
                hybrid_compare_func = generate_compare_function_hybrid(query=query,
                                                                       embedding_model=embedding_model,
                                                                       docs=filtered_docs,
                                                                       alpha=hybrid_alpha)
                key_func = functools.cmp_to_key(hybrid_compare_func)
                filtered_docs = sorted(filtered_docs, key=key_func)
        except Exception as e:
            import traceback
            logger.error(f"Exception in tool: (def _get_article_relevant_documents): {e}", traceback.format_exc())
            pass
            
        if len(filtered_docs)> k:
            filtered_docs = filtered_docs[:k]

        if  "shuffle" in experiment:
            random.shuffle(filtered_docs)
            
        output = document_separator.join(
            format_document(doc, article_meta_data, author_data, document_prompt,
                            experiment = experiment) for doc in filtered_docs
        )
        return ServiceResponse(status=ServiceExecStatus.SUCCESS,
                           content=output)
    except Exception as e:
        return ServiceResponse(status=ServiceExecStatus.ERROR,
                           content=e)


def create_article_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    article_meta_data:dict,
    author_data:dict,
    *,
    embedding_model: Any = None,
    experiment:list = [], # default/shuffle/false cite
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
    filter_keys: list = [
        "topic", "big_name", "write_topic"
    ],
    max_search:int = 5,
    big_name_list:list = [],
    interested_topics:List[str] = [],
    research_topic:str ="",
    hybrid_alpha:float=0.5
) :
    """Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.

    Returns:
        Tool class to pass to an agent
    """
   
    return GraphServiceFactory.get(
        _get_article_relevant_documents,
        name=name,
        description=description,
        retriever=retriever,
        article_meta_data=article_meta_data,
        author_data=author_data,
        document_prompt=document_prompt,
        document_separator=document_separator,
        experiment = experiment,
        filter_keys = filter_keys,
        max_search = max_search,
        big_name_list = big_name_list,
        interested_topics = interested_topics,
        research_topic = research_topic,
        embedding_model = embedding_model,
        hybrid_alpha = hybrid_alpha
    )
