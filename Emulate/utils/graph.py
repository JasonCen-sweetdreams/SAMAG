import torch
from torch_geometric.data import HeteroData
from langchain_core.documents import Document
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

def build_article_graph(
    article_meta: Dict[str, Any],
    author_meta:  Dict[str, Any],
    docs:         List[Document]
) -> Tuple[HeteroData, Dict[int,str], Dict[str,int], Dict[int,Document]]:
    """
    构建一个 HeteroData 图，包含两个节点类型 'article' 和 'author'，
    以及 'cites' 和 'writes' 两种边。

    返回：
      - data:         构建好的 HeteroData
      - node_id_to_title: {article_node_id -> title}
      - title_to_node_id: {title -> article_node_id}
      - index_to_doc: {article_node_id -> Document}
    """
    data = HeteroData()

    # 1) 构建 article 节点映射
    titles = list(article_meta.keys())
    title_to_node_id = {t: i for i, t in enumerate(titles)}
    node_id_to_title = {i: t for t, i in title_to_node_id.items()}
    data['article'].num_nodes = len(titles)

    # 2) 构建 author 节点映射
    author_ids = list(author_meta.keys())
    author_to_node_id = {aid: i for i, aid in enumerate(author_ids)}
    node_id_to_author = {i: aid for aid, i in author_to_node_id.items()}
    data['author'].num_nodes = len(author_ids)

    # 3) 添加 article -> article 引用边（cites）
    src, dst = [], []
    for title, meta in article_meta.items():
        u = title_to_node_id[title]
        for cited in meta.get('cited_articles', []):
            if cited in title_to_node_id:
                v = title_to_node_id[cited]
                src.append(u)
                dst.append(v)
    if src:
        data['article', 'cites', 'article'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 4) 添加 author -> article 写作边（writes）
    src, dst = [], []
    for title, meta in article_meta.items():
        v = title_to_node_id[title]
        for aid in meta.get('author_ids', []):
            if aid in author_to_node_id:
                u = author_to_node_id[aid]
                src.append(u)
                dst.append(v)
    if src:
        data['author', 'writes', 'article'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 5) 建立 node_idx -> Document 的映射
    index_to_doc: Dict[int, Document] = {}
    # docs 是 DirectoryArticleLoader.load() 返回的 list[Document]
    # 每个 doc.metadata['title'] 对应 article_meta 的 key
    for doc in docs:
        t = doc.metadata.get('title')
        if t in title_to_node_id:
            idx = title_to_node_id[t]
            index_to_doc[idx] = doc

    return data, node_id_to_title, title_to_node_id, index_to_doc


def build_social_graph(
    social_member_data: pd.DataFrame,
    docs: List[Document]
) -> Tuple[HeteroData, Dict[int, Document]]:
    """
    构建社交环境的异构图 HeteroData，包含两类节点：'user' 和 'tweet'；
    以及四种有向边：
      - ('user','follows','user')，对应 social_member_data['follow'] 列表；
      - ('user','friends','user')，对应 social_member_data['friend'] 列表；
      - ('user','tweets','tweet')，发布新 tweet；
      - ('user','retweets','tweet')，用户转发/回复 tweet；
    返回：
      - data: HeteroData 图
      - idx_to_doc: {tweet_node_id -> Document} 的映射，以便检索后返回 Document。
    """

    data = HeteroData()

    # ————————————
    # 1) 添加 'user' 和 'tweet' 节点数量
    # ————————————
    num_users = int(social_member_data.shape[0])
    # 先统计所有 docs 中的 tweet_idx 最大值，确定总 tweet 节点数
    tweet_indices = [int(doc.metadata["tweet_idx"]) for doc in docs]
    max_tweet_idx = max(tweet_indices) if tweet_indices else -1
    num_tweets = max_tweet_idx + 1  # 假设 tweet_idx 从0开始连续

    data['user'].num_nodes = num_users
    data['tweet'].num_nodes = num_tweets

    # ————————————
    # 2) 构建 user->user 的 follows 边
    # ————————————
    follows_src = []
    follows_dst = []
    # 假设 social_member_data 有列 "user_index", "follow"（列表），"friend"（列表）
    # user_index 本身等于 row.index
    for row in social_member_data.itertuples():
        u = int(row.user_index)
        for v in row.follow:  # 每个 v 都是 int 型的 user_index
            # 添加单向边 u -> v
            follows_src.append(u)
            follows_dst.append(int(v))

    if follows_src:
        edge_index = torch.tensor([follows_src, follows_dst], dtype=torch.long)
        data['user', 'follows', 'user'].edge_index = edge_index

    # ————————————
    # 3) 构建 user->user 的 friends 边
    # ————————————
    friends_src = []
    friends_dst = []
    for row in social_member_data.itertuples():
        u = int(row.user_index)
        for v_ in row.friend:
            v = int(v_)
            # 添加 u->v
            friends_src.append(u)
            friends_dst.append(v)
            # 添加 v->u（保证双向）
            friends_src.append(v)
            friends_dst.append(u)

    if friends_src:
        edge_index = torch.tensor([friends_src, friends_dst], dtype=torch.long)
        data['user', 'friends', 'user'].edge_index = edge_index


    # ————————————
    # 4) 构建 user->tweet 的 tweets（原创）和 retweets（转发/回复）边
    # ————————————
    tweets_src = []
    tweets_dst = []
    retweets_src = []
    retweets_dst = []
    for doc in docs:
        # doc.metadata 中至少包含：'tweet_idx'、'user_index'、'action'、'origin_tweet_idx'
        t_idx = int(doc.metadata["tweet_idx"])
        u_idx = int(doc.metadata["user_index"])
        action = doc.metadata.get("action", "tweet").lower()
        if action == "tweet":
            # 原创：user -> tweet
            tweets_src.append(u_idx)
            tweets_dst.append(t_idx)
        else:
            # 转发或回复：user -> tweet；虽然源 tweet 放在 origin_tweet_idx，但这里只要把用户映射到该 tweet 上
            retweets_src.append(u_idx)
            retweets_dst.append(t_idx)

    if tweets_src:
        data['user', 'tweets', 'tweet'].edge_index = torch.tensor(
            [tweets_src, tweets_dst], dtype=torch.long
        )
    if retweets_src:
        data['user', 'retweets', 'tweet'].edge_index = torch.tensor(
            [retweets_src, retweets_dst], dtype=torch.long
        )

    # ————————————
    # 5) 构建 tweet_idx -> Document 的映射，便于最后检索后返回 Document
    # ————————————
    idx_to_doc: Dict[int, Document] = {}
    for doc in docs:
        t_idx = int(doc.metadata["tweet_idx"])
        idx_to_doc[t_idx] = doc

    return data, idx_to_doc


def build_movie_graph(
        users_data: np.ndarray,
        movies_data: np.ndarray,
        ratings_data: np.ndarray,
) -> HeteroData:
    """
    根据持久化的用户、电影、评分数据构建一个异构图。
    Nodes:
    - User (e.g., 'u_1')
    - Movie (e.g., 'm_34')
    - Genre (e.g., 'g_Comedy')
    Edges:
    - (User) -[RATED]-> (Movie)  (带有评分和时间戳属性)
    - (Movie) -[HAS_GENRE]-> (Genre)
    """
    data = HeteroData()

    # 1. 节点ID映射
    # 创建从原始ID到从0开始的连续整数索引的映射
    user_ids = np.unique(users_data[:, 0].astype(int))
    user_mapping = {token: i for i, token in enumerate(user_ids)}

    movie_ids = np.unique(movies_data[:, 0].astype(int))
    movie_mapping = {token: i for i, token in enumerate(movie_ids)}

    all_genres = sorted(list(set(genre for genres in movies_data[:, 2] for genre in genres.split('|'))))
    genre_mapping = {token: i for i, token in enumerate(all_genres)}
    
    mapping = {'user': user_mapping, 'movie': movie_mapping, 'genre': genre_mapping}

    # 2. 节点特征编码
    # 用户特征 (Gender, Age, Occupation - 进行独热编码)
    user_df = pd.DataFrame(users_data, columns=['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code', 'Timestamp'])
    user_df['Age'] = user_df['Age'].astype('category')
    user_df['OccupationID'] = user_df['OccupationID'].astype('category')
    
    gender_one_hot = pd.get_dummies(user_df['Gender'], prefix='gender').astype(np.float32)
    age_one_hot = pd.get_dummies(user_df['Age'], prefix='age').astype(np.float32)
    occupation_one_hot = pd.get_dummies(user_df['OccupationID'], prefix='job').astype(np.float32)
    
    user_features = pd.concat([gender_one_hot, age_one_hot, occupation_one_hot], axis=1)
    
    # 按映射顺序排序特征
    sorted_user_features = torch.tensor(user_features.reindex(user_ids).values, dtype=torch.float)
    data['user'].x = sorted_user_features

    # 电影特征 (Genres - 进行Multi-hot编码)
    movie_genres = [row[2].split('|') for row in movies_data]
    mlb = MultiLabelBinarizer(classes=all_genres)
    movie_genre_features = torch.tensor(mlb.fit_transform(movie_genres), dtype=torch.float)
    data['movie'].x = movie_genre_features
    
    # 类型节点没有特征
    data['genre'].num_nodes = len(all_genres)

    # 3. 边索引和边属性
    # 用户 -> 电影 (rated)
    rated_src = [user_mapping[uid] for uid in ratings_data[:, 0].astype(int)]
    rated_dst = [movie_mapping[mid] for mid in ratings_data[:, 1].astype(int)]
    data['user', 'rated', 'movie'].edge_index = torch.tensor([rated_src, rated_dst], dtype=torch.long)
    data['user', 'rated', 'movie'].edge_attr = torch.tensor(ratings_data[:, 2].astype(float), dtype=torch.float).unsqueeze(1) # 评分作为边属性

    # 电影 -> 用户 (rev_rated - 反向边，便于GNN)
    data['movie', 'rev_rated', 'user'].edge_index = torch.tensor([rated_dst, rated_src], dtype=torch.long)
    data['movie', 'rev_rated', 'user'].edge_attr = torch.tensor(ratings_data[:, 2].astype(float), dtype=torch.float).unsqueeze(1)

    # 电影 -> 类型 (has_genre)
    genre_src, genre_dst = [], []
    for movie_row in movies_data:
        movie_id = int(movie_row[0])
        genres = movie_row[2].split('|')
        for genre in genres:
            genre_src.append(movie_mapping[movie_id])
            genre_dst.append(genre_mapping[genre])
    data['movie', 'has_genre', 'genre'].edge_index = torch.tensor([genre_src, genre_dst], dtype=torch.long)
    
    print("HeteroData knowledge graph built successfully.")
    return data, mapping