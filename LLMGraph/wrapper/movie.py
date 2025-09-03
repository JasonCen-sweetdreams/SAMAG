from typing import Dict, Optional,Sequence, Union
from agentscope.agents import AgentBase
from loguru import logger
from .base import BaseAgentWrapper
from agentscope.message import Msg, PlaceholderMessage
import time
from LLMGraph.utils.str_process import remove_before_first_space

from datetime import datetime,date
import random

class MovieAgentWrapper(BaseAgentWrapper):
    
    def __init__(self, 
                 name: Union[str,int], 
                 agent:AgentBase,
                 manager_agent:AgentBase,
                 max_retrys: int = 3, 
                 max_tool_iters: int = 2, 
                 **kwargs) -> None:
        
        """tools: 需要已经to_dist后的agent"""
       
        super().__init__(name, agent, manager_agent, max_retrys, max_tool_iters, **kwargs)


    def reply(self, message:Msg = None) -> Msg:
        available_func_names = ["get_movie_watch_plan",
                                "rate_movie_process",
                                "propose_movie",
                                "generate_initial_thoughts",
                                "participate_in_discussion"]
        func_name = message.get("func")
        assert func_name in available_func_names, f"{func_name} not in {available_func_names}"
        func = getattr(self,func_name)
        kwargs = message.get("kwargs",{})
        func_res = func(**kwargs)
        return func_res
        

    def propose_movie(self, cur_time: str):
        genres = self.choose_genre().content
        filter = {"interested_genres": genres}
        self.call_manager_func("update", kwargs=filter)
        searched_info = self.__get_movie_search_batch()
        res_msg = self.call_agent_func(
            "propose_movie",
            kwargs={ 
                "movie_description": searched_info,
                "group_genres": genres
            }
        )
        res = res_msg.content
        return Msg(content=res, name=self.name, role="assistant")


    def generate_initial_thoughts(self, movie_info: dict):
        role_description = self.call_manager_func("get_watcher_infos", kwargs={"watcher_id": self.name}).content
        res_msg = self.call_agent_func(
            "discuss_movie",
            kwargs={
                "role_description": role_description,
                "movie_title": movie_info.get("title", ""),
                "discussion_context": ""
            }
        )
        res = res_msg.content
        return Msg(content=res, name=self.name, role="assistant")


    def participate_in_discussion(self, movie_info: dict, discussion_context: str):
        role_description = self.call_manager_func("get_watcher_infos", kwargs={"watcher_id": self.name}).content
        res_msg = self.call_agent_func(
            "discuss_movie",
            kwargs={
                "role_description": role_description,
                "movie_title": movie_info.get("title", ""),
                "discussion_context": discussion_context
            }
        )
        res = res_msg.content
        return Msg(content=res, name=self.name, role="assistant")


    def get_movie_watch_plan(self, 
                             task, 
                             len_v:int = 50,
                             use_default_plan:bool = True,
                             requirement:str = "You should watch a lot of movies and rate them."):
        if use_default_plan:
            plans = [
            (random.randint(1,5),
             random.randint(1,5)) for i in range(len_v)
        ] # default plan
            return self.call_agent_func("update_watch_plan",
                                      kwargs={"plans":plans})
            
        role_description = self.call_manager_func("get_watcher_infos", 
                                                  kwargs={
                                                "watcher_id":self.name,
                                                "first_person": True}).content
        
        return self.call_agent_func("get_movie_watch_plan", 
                                      kwargs={"task":task, 
                                              "requirement":requirement,
                                              "role_description":role_description}
                                    )


    def rate_movie_process(self,
                             cur_time: str,
                             discussion_context: Optional[str] = None,
                             existing_reviews_context: Optional[str] = None,
                             pre_selected_movie: Optional[dict] = None):
        self.call_agent_func("watch_step").content
        cur_time_date = datetime.strptime(cur_time, "%Y-%m-%d").date()
        print((f"[{self.name}] Agent is in Group? {pre_selected_movie is not None}"))
        # 场景一：作为群组成员
        if pre_selected_movie:
            movie_searched_infos = f"Movie to review: {pre_selected_movie['title']}\nReason for selection: {pre_selected_movie['reason']}"
            print("Agent in group processing rating...")
            ratings = self.__rate_movie(
                searched_info=movie_searched_infos,
                cur_time=cur_time_date,
                discussion_context=discussion_context,
                existing_reviews_context=existing_reviews_context
            )
            print((f"[{self.name}] Agent in group is returning {len(ratings)} ratings: {ratings}"))
            return Msg(content=ratings, name=self.name, role="assistant")

        # 场景二：作为独行侠
        # 1. 电影发现
        genres = self.choose_genre().content
        filter_kwargs = {"interested_genres": genres, "watched_movie_ids": self.call_agent_func("get_watched_movie_ids").content}
        self.call_manager_func("update", kwargs=filter_kwargs)
        movie_searched_infos = self.__get_movie_search_batch()
        print(f"[{self.name}] agent searched: {movie_searched_infos}")
        
        # 2. 从搜索结果中选择电影
        role_description = self.call_manager_func(
            "get_watcher_infos", 
            kwargs={"watcher_id": self.name, "first_person": False}
        ).content

        selected_movies_msg = self.call_agent_func("select_movies_from_search",
                                                   kwargs={"searched_info": movie_searched_infos,
                                                           "role_description": role_description})
        selected_movies = selected_movies_msg.content
        print(f"[{self.name}] agent selected: {selected_movies}")
        if not selected_movies:
            return Msg(content=[], name=self.name, role="assistant")

        all_ratings = []
        # 3 & 4. 逐一洞察舆论并生成评论
        for movie_to_rate in selected_movies:
            
            try:
                reviews_msg = self.call_manager_func(
                    "call_tool_func",
                    kwargs={ "function_call_msg": [{"tool": "GetReviewsForMovie", "tool_input": {"movie_id": movie_to_rate['movie_id']}}] }
                )
                public_reviews = reviews_msg.content[0][1].content['reviews']
            except Exception:
                public_reviews = "No public reviews could be retrieved for this movie."

            # 4. 生成评论
            ratings = self.__rate_movie(
                searched_info=f"Movie to review: {movie_to_rate['title']}",
                cur_time=cur_time_date,
                interested_genres=genres,
                discussion_context=None,
                existing_reviews_context=public_reviews
            )
            all_ratings.extend(ratings)
        print(f"Agent [{self.name}] is returning {len(all_ratings)} ratings: {all_ratings}")
        return Msg(content=all_ratings, name=self.name, role="assistant")


    def choose_genre(self) -> Msg:
        
        movie_genres_available = self.call_manager_func("get_movie_types").content
        movie_description = self.call_manager_func("get_movie_description").content
        role_description = self.call_manager_func("get_watcher_infos",
                                                  kwargs={"watcher_id":self.name,
                                                    "first_person": False}).content
        return self.call_agent_func(func_name="choose_genre",
                        kwargs={
                            "role_description":role_description,
                            "movie_description":movie_description,
                            "movie_genres_available":movie_genres_available,
                        })
    
        
    def __watch_movie_batch(self,
                        cur_time: date,
                        max_retry: int = 5, # 每个rate最多retry两次
                        filter :dict = {},
                        discussion_context: Optional[str] = ""
                        ):
        """return a list of movie ratings 

        Args:
            movie_manager (MovieManager): _description_
            tools (list, optional): _description_. Defaults to [].
            min_rate (int, optional): _description_. Defaults to 20.
        """
        ratings_all = []
        retry_num = 0
        
        min_rate = self.call_agent_func("get_online_watch_plan").content
        if min_rate is None: return []
        assert isinstance(min_rate,int)
        
        if self.call_manager_func("get_docs_len").content == 0:return []
       
        available_movie_number = self.call_manager_func("get_movie_available_num",
                                                        kwargs={"watched_movie_ids":filter.get("watched_movie_ids",
                                                                                         [])}).content
        
        while (retry_num < max_retry \
            and (len(ratings_all) ) < min_rate\
            and available_movie_number > 0 ):
           
            movie_searched_infos = self.__get_movie_search_batch()
            min_rate_round_ = min_rate if min_rate < available_movie_number \
                else available_movie_number
            
            ratings = self.__rate_movie(
                                    online=True,
                                    searched_info = movie_searched_infos,
                                    cur_time = cur_time,
                                    min_rate_round = min_rate_round_,
                                    max_rate = min_rate_round_ +3,
                                    movie_filter = filter)
            ratings_all.extend(ratings)
            retry_num +=1
        
        return ratings_all
    
    def __get_movie_search_batch(self):
        movie_description = self.call_manager_func("get_movie_description").content

        role_description = self.call_manager_func("get_watcher_infos",
                                    kwargs={"watcher_id":self.name}).content

        agent_msgs = self.call_agent_get_prompt(
                    func_name="get_movie_search_batch",
                            kwargs={
                                "role_description":role_description,
                                "movie_description":movie_description
                            }).content

        
        response = self.step(agent_msgs=agent_msgs,
                    use_tools=True,
                    return_intermediate_steps=True,
                    return_tool_exec_only=True)
        
        intermediate_steps = response.get("intermediate_steps",[])
        
        template = """
Here's some information you get from online website:

{searched_infos}

The end of online information.       
"""
        searched_infos = ""
        for intermediate_step in intermediate_steps:
            action, observation = intermediate_step
            searched_infos += observation.get("result","") + "\n"
        return template.format(searched_infos = 
                               searched_infos)
    
    def __rate_movie(self,
                     searched_info:str,
                     cur_time: Union[date,str],
                     interested_genres: list = [],
                     discussion_context: Optional[str] = None,
                     existing_reviews_context: Optional[str] = None):
        if isinstance(cur_time,str):
            cur_time_str = cur_time
        else:
            cur_time_str = cur_time.strftime("%Y-%m-%d")

        role_description = self.call_manager_func("get_watcher_infos", kwargs={"watcher_id":self.name, "first_person":False}).content
       
        ratings = self.call_agent_func(func_name="rate_movie",
                                kwargs={
                                    "role_description":role_description,
                                    "searched_info":searched_info,
                                    "interested_genres": interested_genres,
                                    "discussion_context": discussion_context or "",
                                    "existing_reviews_context": existing_reviews_context or ""
                                }).content
        
        filtered_ratings = []
        for rating in ratings:
            # 注意：原代码中 filter_rating_movie 有一个 online 参数，这里简化了调用
            filter_rating_response = self.call_manager_func("filter_rating_movie", kwargs={"movie_rating":rating}).content
            
            if filter_rating_response:
                filter_rating_response['timestamp'] = cur_time_str
                filter_rating_response['agent_id'] = self.name
                self.call_agent_func("update_rating", kwargs={"rating": filter_rating_response})
                filtered_ratings.append(filter_rating_response)
        return filtered_ratings
    
    # def __rate_movie(self,
    #                  online:bool,
    #                 searched_info:str,
    #                 cur_time: Union[date,str],
    #                 min_rate_round:int = 5, # 最少看几部电影
    #                 max_rate: int = 30, # 最多看几部电影   
    #                 movie_filter:dict ={},
    #                 discussion_context: Optional[str] = None
    #                 ):
    #     if isinstance(cur_time,str):
    #         cur_time_str = cur_time
    #         cur_time = datetime.strptime(cur_time_str, "%Y-%m-%d").date()
    #     else:
    #         cur_time_str = cur_time.strftime("%Y-%m-%d")
    #     role_description = self.call_manager_func("get_watcher_infos", 
    #                                               kwargs={
    #                                             "watcher_id":self.name,
    #                                             "first_person":False
    #                                             }).content
       
    #     ratings = self.call_agent_func(func_name="rate_movie",
    #                     kwargs={
    #                         "role_description":role_description,
    #                         "searched_info":searched_info,
    #                         "min_rate_round":min_rate_round,
    #                         "max_rate":max_rate,
    #                         "interested_genres":movie_filter.get("interested_genres",[]),
    #                         "discussion_context": discussion_context or ""
    #                     }).content
    #     filtered_ratings = []
    #     for rating in ratings:
    #         filter_rating_response = self.call_manager_func("filter_rating_movie",
    #                                     kwargs={
    #                                         "movie_rating":rating,
    #                                         "online": online
    #                                     }).content
            
    #         if filter_rating_response != {}:
                
    #             filter_rating_response['timestamp'] = cur_time_str
    #             self.call_agent_func("update_rating",
    #                 kwargs = {"rating":filter_rating_response})
    #             filtered_ratings.append(filter_rating_response)
    #     return filtered_ratings


    def __watch_movie_offline(self,
                            cur_time:date,
                            filter: dict = {},
                            max_movie_number = 20):
        min_rate_round = self.call_agent_func("get_offline_watch_plan").content
        if min_rate_round is None: return []
        assert isinstance(min_rate_round,int)
        searched_movie_info = self.call_manager_func(
            "get_offline_movie_info",
            kwargs= {
                "filter":filter,
                "max_movie_number":max_movie_number
            }
        ).content
        
        if searched_movie_info == "":return []

        template = """
Here's some information you get from online website:

{searched_infos}

The end of online information.       
"""
        searched_movie_info = template.format(searched_infos = searched_movie_info)    
        return self.__rate_movie(
                        online=False,
                        searched_info =searched_movie_info,
                        cur_time=cur_time,
                        min_rate_round = min_rate_round,
                        max_rate = min_rate_round+3,
                        movie_filter=filter
                        )
        
    def get_agent_memory_msgs(self):
        return self.call_agent_func(
            "get_short_memory"
        )