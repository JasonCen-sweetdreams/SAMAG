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
    Build a HeteroData graph with two node types 'article' and 'author',
    and two edge types 'cites' and 'writes'.

    Returns:
      - data: the constructed HeteroData
      - node_id_to_title: {article_node_id -> title}
      - title_to_node_id: {title -> article_node_id}
      - index_to_doc: {article_node_id -> Document}
    """
    data = HeteroData()

    # 1) Build article node mapping
    titles = list(article_meta.keys())
    title_to_node_id = {t: i for i, t in enumerate(titles)}
    node_id_to_title = {i: t for t, i in title_to_node_id.items()}
    data['article'].num_nodes = len(titles)

    # 2) Build author node mapping
    author_ids = list(author_meta.keys())
    author_to_node_id = {aid: i for i, aid in enumerate(author_ids)}
    node_id_to_author = {i: aid for aid, i in author_to_node_id.items()}
    data['author'].num_nodes = len(author_ids)

    # 3) Add article -> article citation edges ('cites')
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

    # 4) Add author -> article writing edges ('writes')
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

    # 5) Build mapping from node_idx -> Document
    index_to_doc: Dict[int, Document] = {}
    # docs: the list[Document] returned by DirectoryArticleLoader.load()
    # Each doc.metadata['title'] corresponds to a key in article_meta
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
    Build a heterogeneous social HeteroData graph with two node types:
      'user' and 'tweet';
    and four directed edge types:
      - ('user','follows','user'): from social_member_data['follow'] lists
      - ('user','friends','user'): from social_member_data['friend'] lists
      - ('user','tweets','tweet'): user posts a new tweet
      - ('user','retweets','tweet'): user retweets/replies to a tweet

    Returns:
      - data: the HeteroData graph
      - idx_to_doc: mapping {tweet_node_id -> Document} for retrieval
    """

    data = HeteroData()

    # 1) Add node counts for 'user' and 'tweet'
    num_users = int(social_member_data.shape[0])
    tweet_indices = [int(doc.metadata["tweet_idx"]) for doc in docs]
    max_tweet_idx = max(tweet_indices) if tweet_indices else -1
    num_tweets = max_tweet_idx + 1

    data['user'].num_nodes = num_users
    data['tweet'].num_nodes = num_tweets

    # 2) Build user->user 'follows' edges
    follows_src = []
    follows_dst = []
    for row in social_member_data.itertuples():
        u = int(row.user_index)
        for v in row.follow:
            follows_src.append(u)
            follows_dst.append(int(v))

    if follows_src:
        edge_index = torch.tensor([follows_src, follows_dst], dtype=torch.long)
        data['user', 'follows', 'user'].edge_index = edge_index

    # 3) Build user->user 'friends' edges
    friends_src = []
    friends_dst = []
    for row in social_member_data.itertuples():
        u = int(row.user_index)
        for v_ in row.friend:
            v = int(v_)
            friends_src.append(u)
            friends_dst.append(v)
            friends_src.append(v)
            friends_dst.append(u)

    if friends_src:
        edge_index = torch.tensor([friends_src, friends_dst], dtype=torch.long)
        data['user', 'friends', 'user'].edge_index = edge_index


    # 4) Build user->tweet 'tweets' (original) and 'retweets' (retweet/reply) edges
    tweets_src = []
    tweets_dst = []
    retweets_src = []
    retweets_dst = []
    for doc in docs:
        t_idx = int(doc.metadata["tweet_idx"])
        u_idx = int(doc.metadata["user_index"])
        action = doc.metadata.get("action", "tweet").lower()
        if action == "tweet":
            tweets_src.append(u_idx)
            tweets_dst.append(t_idx)
        else:
            # retweet or reply: user -> tweet; although the source tweet is in origin_tweet_idx,
            # we only need to connect the user to the (current) tweet here
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

    # 5) Build mapping tweet_idx -> Document for retrieval
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
    Build a heterogeneous graph from persisted users, movies, and ratings.

    Nodes:
    - User (e.g., 'u_1')
    - Movie (e.g., 'm_34')
    - Genre (e.g., 'g_Comedy')

    Edges:
    - (User) -[RATED]-> (Movie)  (with rating and timestamp attributes)
    - (Movie) -[HAS_GENRE]-> (Genre)
    """
    data = HeteroData()

    # 1. Node ID mappings
    user_ids = np.unique(users_data[:, 0].astype(int))
    user_mapping = {token: i for i, token in enumerate(user_ids)}

    movie_ids = np.unique(movies_data[:, 0].astype(int))
    movie_mapping = {token: i for i, token in enumerate(movie_ids)}

    all_genres = sorted(list(set(genre for genres in movies_data[:, 2] for genre in genres.split('|'))))
    genre_mapping = {token: i for i, token in enumerate(all_genres)}
    
    mapping = {'user': user_mapping, 'movie': movie_mapping, 'genre': genre_mapping}

    # 2. Node feature encoding
    user_df = pd.DataFrame(users_data, columns=['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code', 'Timestamp'])
    user_df['Age'] = user_df['Age'].astype('category')
    user_df['OccupationID'] = user_df['OccupationID'].astype('category')
    
    gender_one_hot = pd.get_dummies(user_df['Gender'], prefix='gender').astype(np.float32)
    age_one_hot = pd.get_dummies(user_df['Age'], prefix='age').astype(np.float32)
    occupation_one_hot = pd.get_dummies(user_df['OccupationID'], prefix='job').astype(np.float32)
    
    user_features = pd.concat([gender_one_hot, age_one_hot, occupation_one_hot], axis=1)
    
    sorted_user_features = torch.tensor(user_features.reindex(user_ids).values, dtype=torch.float)
    data['user'].x = sorted_user_features

    movie_genres = [row[2].split('|') for row in movies_data]
    mlb = MultiLabelBinarizer(classes=all_genres)
    movie_genre_features = torch.tensor(mlb.fit_transform(movie_genres), dtype=torch.float)
    data['movie'].x = movie_genre_features
    
    data['genre'].num_nodes = len(all_genres)

    # 3. Edge indices and edge attributes
    # User -> Movie (rated)
    rated_src = [user_mapping[uid] for uid in ratings_data[:, 0].astype(int)]
    rated_dst = [movie_mapping[mid] for mid in ratings_data[:, 1].astype(int)]
    data['user', 'rated', 'movie'].edge_index = torch.tensor([rated_src, rated_dst], dtype=torch.long)
    data['user', 'rated', 'movie'].edge_attr = torch.tensor(ratings_data[:, 2].astype(float), dtype=torch.float).unsqueeze(1) # 评分作为边属性

    # Movie -> User
    data['movie', 'rev_rated', 'user'].edge_index = torch.tensor([rated_dst, rated_src], dtype=torch.long)
    data['movie', 'rev_rated', 'user'].edge_attr = torch.tensor(ratings_data[:, 2].astype(float), dtype=torch.float).unsqueeze(1)

    # Movie -> Genre (has_genre)
    genre_src, genre_dst = [], []
    for movie_row in movies_data:
        movie_id = int(movie_row[0])
        genres = movie_row[2].split('|')
        for genre in genres:
            genre_src.append(movie_mapping[movie_id])
            genre_dst.append(genre_mapping[genre])
    data['movie', 'has_genre', 'genre'].edge_index = torch.tensor([genre_src, genre_dst], dtype=torch.long)
    
    return data, mapping