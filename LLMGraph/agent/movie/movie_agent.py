from agentscope.message import Msg, PlaceholderMessage
import asyncio

from loguru import logger
from LLMGraph.prompt.movie import movie_prompt_registry

from LLMGraph.output_parser import movie_output_parser_registry
from typing import Any, List, Optional, Tuple, Union,Dict, Sequence
from pydantic import root_validator

from pandas._libs.tslibs import Timestamp
from datetime import datetime, date, timedelta

from LLMGraph.memory import movie_memory_registry, MovieMemory
from LLMGraph.manager import MovieManager

import re
import random
import copy

from langchain_core.runnables import Runnable
from .. import agent_registry
from openai import RateLimitError,AuthenticationError

from time import sleep
import numpy as np
from LLMGraph.agent.base_agent import BaseGraphAgent

from agentscope.memory import TemporaryMemory

def remove_before_first_space(s:str):
    # 分割字符串，最大分割次数设为1
    parts = s.split(' ', 1)
    # 如果字符串中有空格，则返回第一个空格后的所有内容
    if len(parts) > 1:
        return parts[1].replace("\"","")
    else:
        # 如果没有空格，返回整个字符串
        return s.replace("\"","")


@agent_registry.register("movie_agent")
class MovieAgent(BaseGraphAgent):

    movie_memory:MovieMemory
    
    def __init__(self, 
                name:str,
                infos:dict,
                agent_configs:dict,
                short_memory_config:dict = {},
                rating_counts: dict ={},
                ratings:dict ={},
                ):
        
       
        memory_config = agent_configs.get('memory')
        memory_config["name"] = name

       
        memory_config["rating_counts"] = rating_counts
        memory_config["ratings"] = ratings
        
        self.movie_memory = movie_memory_registry.build(**memory_config)
        self.mode = "choose_genre"
        # random default plan
        self.plans = [
            (random,1) for i in range(50)
        ] # default plan
        self.plan_ptr = -1
        self.infos = infos
        self.short_memory = TemporaryMemory(short_memory_config)
        self.rate_output_format = """
- If you want to give rating of a certain movie, you should give your rating in this format:
    Movie: (The name of the movie)
    Thought: (Your view on this movie and the reason why you gave this rating score of movie)
    Rating: (Your rating of this movie, from 0 to 5)
    Finish: (If you want to watch more movies, set this value to False. Else set to True)
        
Respond:"""

        prompt = movie_prompt_registry.build(self.mode)
        output_parser = movie_output_parser_registry.build(self.mode)
        agent_llm_config = agent_configs.get('llm')

        super().__init__(name,
                         prompt_template=prompt,
                         output_parser=output_parser,
                         model_config_name=agent_llm_config["config_name"],
                         use_memory=False # ban default memory
                         )
        
        
        
    
    def reset_state(self,
                    mode = "choose_genre"):
       
        if self.mode == mode: return
        self.mode = mode
        
        prompt = movie_prompt_registry.build(mode)
        output_parser = movie_output_parser_registry.build(mode)
        self.prompt_template = prompt
        self.output_parser = output_parser
    

    def propose_movie(self, movie_description: str, group_genres: list):
        self.reset_state("propose_movie")
        prompt_inputs = {
            "movie_description": movie_description,
            "group_genres": ", ".join(group_genres)
        }
        response = self.step(prompt_inputs)
        res = response.content
        # logger.info(f"propose_movie response: {res}")
        try:
            result = res.get("return_values", {})
            return Msg(self.name, role="assistant", content=result)
        except Exception as e:
            logger.warning(f"Exception in propose_movie: {e}\n:response: {res}")
    

    def discuss_movie(self, role_description: str, movie_title: str, discussion_context: str):
        self.reset_state("discuss_movie")
        prompt_inputs = {
            "role_description": role_description,
            "movie_title": movie_title,
            "discussion_context": discussion_context
        }
        
        response = self.step(prompt_inputs)
        try:
            res = response.content.get("return_values", {}).get("speak", "")
            return Msg(self.name, role="assistant", content=res)
        except Exception as e:
            logger.warning(f"Exception in discuss_movie: {e}\n:response: {response}")

    
    def update_watch_plan(self,plans:list=[]):
        self.plans = plans

    def get_movie_watch_plan(self,
                             role_description:str,
                             task:str,
                             requirement:str = "You should watch a lot of movies and rate them."):
        
        self.reset_state("watch_plan")
        prompt_inputs = {
            "role_description": role_description,
            "requirement": requirement,
            "task": task
        }
        response = self.step(prompt_inputs = prompt_inputs)
        try:
            plans = response.content.get("return_values",{}).get("plan",[])
            assert isinstance(plans,list)
            self.plans = plans
        except Exception as e:
            print(e)
          
    
    def get_watched_movie_ids(self):
        watched_movie_ids = self.movie_memory.get_watched_movie_ids()
        return Msg(
            self.name,
            content=watched_movie_ids,
            role="assistant"
        )
    
    def get_watched_movie_names(self):
        watched_movie_names = self.movie_memory.get_watched_movie_names()
        return Msg(
            self.name,
            content=watched_movie_names,
            role="assistant"
        )
    
    def get_movie_search_batch(self,
                               role_description,
                               movie_description,
                            ):
        self.reset_state("watch_movie_batch")
        """return a list of movie ratings 

        Args:
            movie_manager (MovieManager): _description_
            tools (list, optional): _description_. Defaults to [].
            min_rate (int, optional): _description_. Defaults to 20.
        """
        
        instruction = """You should make full use of your tools to get movie information."""
        
        prompt_inputs = {
            "role_description": role_description,
            "instruction": instruction,
            "movie_description": movie_description,
            "memory":self.movie_memory.retrieve_movie_memory()
        }
        

        return prompt_inputs
    

    def rate_movie(self,
                   role_description:str,
                   searched_info:str,
                   min_rate_round:int = 5, # 最少看几部电影
                   max_rate: int = 30, # 最多看几部电影
                   interested_genres: list = [],
                   discussion_context: str = "",
                   existing_reviews_context: str = ""
                   ):
        self.reset_state("rate_movie")
        movie_memory = self.movie_memory.retrieve_movie_memory()
        
        prompt_inputs = {
            "role_description": role_description,
            "movie_description": searched_info,
            "memory": movie_memory,
            "min_rate" : 1, # 硬编码为1，因为每次只评一部电影
            "max_rate" : 1,
            "watched_movie_names": ",".join(self.movie_memory.get_watched_movie_names()),
            "interested_genres": ",".join(interested_genres),
            "discussion_context": discussion_context,
            "existing_reviews_context": existing_reviews_context
        }

        response = self.step(prompt_inputs = prompt_inputs).content
        try:
            ratings = response.get("return_values",{}).get("ratings",[])
        except:
            ratings = []
        return Msg(content=ratings, name=self.name, role="assistant")
       
        
        
    def choose_genre(self,
                     movie_genres_available,
                     role_description:str,
                     movie_description:str
                           ) -> list:
        self.reset_state(mode="choose_genre")
        
        prompt_inputs={
            "role_description": role_description,
            "memory": self.movie_memory.retrieve_movie_memory(),
            "movie_description": movie_description
        }
        
        response = self.step(prompt_inputs)
        interested_genres = []
        try:
            response = response.content.get("return_values",{})
            genres = response.get("output","").split(",")
            
            for genre in genres: 
               for candidate_genre in movie_genres_available:
                   if genre.lower() in candidate_genre.lower() or \
                       candidate_genre.lower() in genre.lower():
                           interested_genres.append(candidate_genre)
                           break
            
        except Exception as e:
            print(e)

        return Msg(content=interested_genres,
                   name=self.name,
                   role = "assistant")
        
    def watch_step(self):
        self.plan_ptr += 1

    def get_offline_watch_plan(self):
        min_rate_round = None # 不rate
        if self.plan_ptr < len(self.plans):
            try:
                min_rate_round = int(self.plans[self.plan_ptr][1])
                if min_rate_round ==0: return Msg(self.name,
                                                 content=min_rate_round,
                                                 role="assistant")
            except:min_rate_round = 1
        
        return Msg(self.name,content=min_rate_round,role="assistant")
    
    def get_online_watch_plan(self):
        min_rate = None
        if self.plan_ptr < len(self.plans):
            try:
                min_rate = int(self.plans[self.plan_ptr][0])
                if min_rate ==0:return Msg(self.name,content=min_rate,role="assistant")
            except:
                min_rate = 5
        return Msg(self.name,content=min_rate,role="assistant")
    
    def update_rating(self,
                      rating):
        self.movie_memory.update_rating(rating)

    def observe(self, messages:Union[Msg]=None):
        if not isinstance(messages,Sequence):
            messages = [messages]
        for message in messages:
            if isinstance(message,PlaceholderMessage):
                message.update_value()
            # if not isinstance(message,Msg):continue
            if message.content == "":
                continue
            if message.name == self.name or message.name == "system":
                self.short_memory.add(messages)
                break
    
    def select_movies_from_search(self, role_description: str, searched_info: str) -> Msg:
        """从搜索到的电影列表中选择1-3部进行评价。"""
        # 假设有这样一个新状态和对应的prompt, parser
        self.reset_state("select_movies") 
        prompt_inputs = {
            "role_description": role_description,
            "searched_info": searched_info,
            "memory": self.movie_memory.retrieve_movie_memory()
        }
        # response = self.step(prompt_inputs)
        # selected = response.content.get("return_values", {}).get("selected_movies", [])

        response_msg = self.step(prompt_inputs)
        
        # --- 健壮性检查 ---
        if response_msg.get("fail", False) or not isinstance(response_msg.content, dict):
            logger.warning(f"Agent [{self.name}] failed to select movies because LLM output parsing failed.")
            logger.debug(f"Raw content that failed parsing: {response_msg.content}")
            selected = [] # 在失败时返回一个安全的空列表
            # raise RuntimeError(f"Failed llm! content: {response_msg.content}")
        else:
            # 只有在成功时才执行原来的逻辑
            selected = response_msg.content.get("return_values", {}).get("selected_movies", [])

        return Msg(self.name, role="assistant", content=selected)