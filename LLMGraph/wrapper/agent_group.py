"""A general dialog agent."""
import random
import time
import re
from typing import Dict, List, Tuple
from loguru import logger
import time

from agentscope.message import Msg
from agentscope.agents import AgentBase
from agentscope.message import Msg,PlaceholderMessage
from LLMGraph.wrapper.article import ArticleAgentWrapper
from LLMGraph.wrapper.social import SocialAgentWrapper
from LLMGraph.wrapper.movie import MovieAgentWrapper


class GroupDiscussAgent(AgentBase):
    """A Moderator to collect values from participants."""

    def __init__(
        self,
        name: str,
        communication_num:int = 5,
        agents:List[AgentBase] = [],
        manager_agent: AgentBase = None,
        to_dist = False
    ) -> None:
        super().__init__(name, to_dist=to_dist)
        self.agents = agents
        self.communication_num = communication_num
        self.manager_agent = manager_agent

    def call_manager_agent_func(self,
                                func_name:str,
                                kwargs:dict ={})->Msg:
        msg = Msg("user",
                content="call_function",
                role="assistant",
                kwargs=kwargs,
                func=func_name,
                )
        return_msg = self.manager_agent(msg)
        return return_msg
    
    def call_agent_func(self,
                        agent:ArticleAgentWrapper,
                        func_name:str,
                        kwargs:dict={}) -> Msg:
        msg = Msg("user",
                content="call_function",
                role="assistant",
                kwargs=kwargs,
                func=func_name
                )
        return_msg = agent(msg)
        if isinstance(return_msg,PlaceholderMessage):
            return_msg.update_value()
        return return_msg


    def communication(self, research_content) -> Msg:
        proposals = []

        first_agent = self.agents[0]
        role_desc_first = self.call_manager_agent_func("get_author_description",
                                                kwargs={"agent_name":first_agent.name}).content
        interested_topics = self.call_agent_func(first_agent, "return_interested_topics").content
        try:
            research_topic = research_content.get("topic", None)
        except:
            print("error", research_content)
        self.call_manager_agent_func("update",
                                     kwargs={"interested_topics":interested_topics,
                                             "research_topic":research_topic})
        # 起始的proposal
        init_proposal_msg = self.call_agent_func(first_agent,
                                                 "proposal_initialization",
                                                 kwargs={
                                                    "role_description": role_desc_first,
                                                    "research_content": research_content,
                                                    "interested_topics": interested_topics
                                                    })
        logger.info(f"init_proposal_msg: {init_proposal_msg.content}")
        if not init_proposal_msg.content:
            return Msg(
                name=self.name,
                role="assistant",
                content="",
            )
        proposals.append({
        "author_id": first_agent.name,
        "role_description": role_desc_first,
        "content": init_proposal_msg.content
        })
        for ag in self.agents:
            ag.observe(init_proposal_msg)
        # 后面的作者提建议
        for idx in range(1, len(self.agents)):
            agent = self.agents[idx]
            role_desc = self.call_manager_agent_func(
                "get_author_description",
                kwargs={"agent_name": agent.name}
            ).content

            suggestion_msg = self.call_agent_func(
                agent,
                "suggestion_generation",
                kwargs={
                    "role_description": role_desc,
                    "proposals": proposals,
                    "research_content": research_content
                }
            )
            logger.info(f"Current agent-{idx}: {agent.name}, suggestion_msg: {suggestion_msg.content}")
            proposals.append({
                "author_id": agent.name,
                "role_description": role_desc,
                "content": suggestion_msg.content
            })
            logger.info(f"suggestion_msg: {suggestion_msg.content}")
            for ag in self.agents:
                ag.observe(suggestion_msg)
        # 汇总建议
        agg_msg = self.call_agent_func(
            first_agent,
            "proposal_generation",
            kwargs={
                "role_description": role_desc_first,
                "proposals": proposals,
                "research_content": research_content
            }
        )

        try:
            research_content_update = agg_msg.content
            assert isinstance(research_content_update,dict)
            # for ag in self.agents:
            #     ag.observe(agg_msg)
        except:
            research_content_update = research_content

        return Msg(
            name=self.name,
            role="assistant",
            content=research_content_update,
        )
        # for idx in range(self.communication_num):
        #     agent = self.agents[idx%len(self.agents)]
        #     role_description = self.call_manager_agent_func("get_author_description",
        #                                             kwargs={"agent_name":agent.name}).content

            
        #     candidate_id_msg = self.call_agent_func(agent, "choose_researcher",
        #                             kwargs={"role_description":role_description,
        #                                     "research_content":research_content})
            
        #     candidate_id = candidate_id_msg.content
        #     role_description_2 = self.call_manager_agent_func("get_author_description",
        #                                     kwargs={"agent_name":candidate_id}).content
            
        #     group_discussion_msg = self.call_agent_func(agent, "group_discuss",
        #                             kwargs={"role_description_1":role_description,
        #                                     "role_description_2":role_description_2,
        #                                     "research_content":research_content,
        #                                     "author_id": candidate_id})
            
        #     for agent in self.agents:
        #         agent.observe(group_discussion_msg)

        #             # print(role_description)
        # research_content_msg = self.call_agent_func(agent,
        #                                             "idea_generation",
        #                                             kwargs={"role_description":role_description,
        #                                         "research_content":research_content})
        # try:
        #     research_content_update = research_content_msg.content
        #     assert isinstance(research_content_update,dict)
        # except:
        #     research_content_update = research_content

        # return Msg(
        #     name=self.name,
        #     role="assistant",
        #     content=research_content_update,
        # )
    

    def write(self, 
              research_content,
              cur_time_str) -> Msg:
        agent_first_author = self.agents[0]

        # interested_topics = self.call_agent_func(agent_first_author, "return_interested_topics").content
        # try:
        #     research_topic = research_content.get("topic", None)
        # except:
        #     print("error", research_content)
        # self.call_manager_agent_func("update",
        #                              kwargs={"interested_topics":interested_topics,
        #                                      "research_topic":research_topic})

        
        research_content = self.call_agent_func(agent_first_author, 
                            "write_process",
                        kwargs={"research_content":research_content}).content
        
        
        # research_content = self.call_agent_func(agent_first_author, 
        #                                             "choose_reason",
        #                                         kwargs={"research_content":research_content,
        #                                                 "cur_time_str":cur_time_str}).content
        if research_content.get("topic") is not None:
            self.call_agent_func(agent_first_author,
                                 "update_interested_topics",
                                     kwargs={
                                         "topics": [research_content.get("topic")]
                                     })

        return Msg(
            name=self.name,
            role="assistant",
            content=research_content,
        )
        
    def reply(self, message: Msg) -> Msg:
        func_name = message.get("func","")
        kwargs = message.get("kwargs",{})
        func = getattr(self,func_name)
        res = func(**kwargs)
        assert isinstance(res,Msg)

        return res
    
class SocialGroupAgent(AgentBase):
    """
    用于将多个 SocialAgentWrapper（对应不同 Twitter 账号）编排成“一领头账号 + 多跟随账号”的群体操作逻辑：
    - 领头账号先调用 twitter_process(cur_time, big_name=True)，得到它本轮要执行的操作列表 leader_actions。
    - 接着把 leader_actions 打包成一条 Msg，依次让各个跟随账号调用 observe(...)，把 leader_actions 写入它们的短期记忆。
    - 最后每个跟随账号再调用 twitter_process(cur_time, big_name=False)，得到它们各自的操作列表 follower_actions。
    - 最终将所有账号的“本轮操作”按账号名汇总成一个 dict，返回给上层。
    """

    def __init__(
        self,
        name: str,
        agents: List[Tuple[SocialAgentWrapper, bool]],
        manager_agent: AgentBase,
        to_dist = False
    ) -> None:
        """
        :param name: 该 Group Agent 的名字
        :param agents: 一个 SocialAgentWrapper 的列表，列表第一个元素会被视为“领头账号（leader）”
        :param manager_agent: 用于提供用户资料、社交网络信息、粉丝数等工具的 Agent
        """
        super().__init__(name, to_dist=to_dist)
        assert len(agents) >= 1, "At least 1 SocialAgentWrapper is needed"
        self.agents = agents
        self.manager_agent = manager_agent

    def call_manager_agent_func(self, func_name: str, kwargs: dict = {}) -> Msg:
        msg = Msg(
            "user",
            content="call_function",
            role="assistant",
            func=func_name,
            kwargs=kwargs
        )
        return self.manager_agent(msg)

    def call_agent_func(
        self,
        agent: SocialAgentWrapper,
        func_name: str,
        kwargs: dict = {}
    ) -> Msg:
        msg = Msg(
            "user",
            content="call_function",
            role="assistant",
            func=func_name,
            kwargs=kwargs
        )
        return_msg = agent(msg)
        if isinstance(return_msg, PlaceholderMessage):
            return_msg.update_value()
        return return_msg

    def communication(self, cur_time: str) -> Msg:
        start_time = time.perf_counter()
        ### 并行1：leader与follower自身行动
        leader_start_time = time.perf_counter()
        leader: SocialAgentWrapper = self.agents[0][0]
        leader_is_big_name: bool = self.agents[0][1]
        followers = self.agents[1:]
        leader_task: Msg = self.call_agent_func(
            leader,
            "twitter_process",
            {"cur_time": cur_time, "big_name": leader_is_big_name}
        )

        follower_start_time = time.perf_counter()
        follower_self_tasks = [
            self.call_agent_func(
                follower,
                "twitter_process",
                {"cur_time": cur_time, "big_name": is_big_name, "in_group": True}
                ) for follower, is_big_name in followers
        ]
        
        leader_actions = leader_task.content or []
        leader_end_time = time.perf_counter()
        ### 并行2：follower群体行动
        follower_group_tasks = [
            self.call_agent_func(
                follower,
                "twitter_process",
                {"cur_time": cur_time, "big_name": is_big_name, "group_context": leader_actions}
            ) for follower, is_big_name in followers
        ]

        final_actions = leader_actions.copy()

        for task in follower_group_tasks:
            actions = task.content or []
            final_actions.extend(actions)

        for task in follower_self_tasks:
            actions = task.content or []
            final_actions.extend(actions[:10])
        follower_end_time = time.perf_counter()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"Current Group size: {len(self.agents)}\ntotal time consume: {total_time}\n \
                    leader time consume: {leader_end_time - leader_start_time}\n \
                    follower total consume: {follower_end_time - follower_start_time}\n \
                    ")
        return Msg(
            name=self.name,
            role="assistant",
            content=final_actions
        )
        

    def communication_old(self, cur_time: str) -> Msg:
        """
        串行化有点严重, 
        :param cur_time: 当前日期字符串，格式为 "YYYY-MM-DD"。
        1. 领头账号（self.agents[0][0]）先调用 twitter_process(cur_time, big_name=True)。
        2. 把领头账号的“本轮操作列表”打包成 Msg，让所有跟随账号先 observe(这条 Msg)，
           以便它们把“leader 的行为”写入短期记忆里。
        3. 然后逐个让每个跟随账号调用 twitter_process(cur_time, big_name=False)，
           得到它们各自的操作列表。
        4. 最终把每个账号的 actions 汇总到一个 dict 里，返回给上层调用者。
        """
        # 1. 领头账号先做一次 twitter_process
        leader: SocialAgentWrapper = self.agents[0][0]
        leader_is_big_name: bool = self.agents[0][1]
        # big_name=True 代表它是领头大号，可以跑 forum_action_bigname 的逻辑
        leader_resp: Msg = self.call_agent_func(
            leader,
            "twitter_process",
            {"cur_time": cur_time, "big_name": leader_is_big_name}
        )
        leader_actions = leader_resp.content or []  # 一定是列表形式

        # 2. 把领头账号的操作打包成一条 Msg，让其他账号 observe
        # leader_msg = Msg(
        #     name=leader.name,
        #     role="assistant",
        #     content=leader_actions
        # )

        # 用一个字典来记录“每个账号本轮实际要执行的操作”
        # group_actions: Dict[str, List[dict]] = {
        #     leader.name: leader_actions
        # }
        group_actions = leader_actions.copy()

        # 3. 对于每个跟随账号（从索引 1 开始），先 observe，再让它调用 twitter_process
        for follower_tuple in self.agents[1:]:
            follower = follower_tuple[0]
            is_big_name = follower_tuple[1]
            # follower.observe(leader_msg) # TODO: 临时注释掉，因为现在每个user的 memory 都是action memory，而不是看到的memory

            # 再让 follower 自己执行 twitter_process
            follower_resp: Msg = self.call_agent_func(
                follower,
                "twitter_process",
                {"cur_time": cur_time, "big_name": is_big_name, "group_context": leader_actions}
            )
            follower_actions = follower_resp.content or []

            # 整理到 group_actions 里
            group_actions.extend(follower_actions)

            follower_resp2: Msg = self.call_agent_func(
                follower,
                "twitter_process",
                {"cur_time": cur_time, "big_name": is_big_name, "in_group": True}
            )
            follower_actions2 = follower_resp2.content or []
            group_actions.extend(follower_actions2[:10])

        return Msg(
            name=self.name,
            role="assistant",
            content=group_actions
        )

    def reply(self, message: Msg) -> Msg:
        func_name = message.get("func", "")
        kwargs = message.get("kwargs", {})
        func = getattr(self, func_name)
        res: Msg = func(**kwargs)
        assert isinstance(res, Msg)
        return res
    

class MovieGroupAgent(AgentBase):
    """
    用于编排电影兴趣小组的Agent。
    负责组织成员进行电影选择、观影后讨论，并收集最终的影评。
    """

    def __init__(
            self,
            name: str,
            agents: List[MovieAgentWrapper] = [],
            manager_agent: AgentBase = None,
            to_dist: bool = False
    ) -> None:
        super().__init__(name, to_dist=to_dist)
        self.agents = agents
        self.manager_agent = manager_agent
    
    def call_manager_agent_func(self,
                                func_name:str,
                                kwargs:dict ={})->Msg:
        msg = Msg("user",
                content="call_function",
                role="assistant",
                kwargs=kwargs,
                func=func_name,
                )
        return_msg = self.manager_agent(msg)
        return return_msg
    
    def call_agent_func(self,
                        agent:ArticleAgentWrapper,
                        func_name:str,
                        kwargs:dict={}) -> Msg:
        msg = Msg("user",
                content="call_function",
                role="assistant",
                kwargs=kwargs,
                func=func_name
                )
        return_msg = agent(msg)
        if isinstance(return_msg,PlaceholderMessage):
            return_msg.update_value()
        return return_msg
    
    def group_activity(self, cur_time: str) -> Msg:
        logger.info(f"Group [{self.name}] starting activity...")

        # 1. 提案 (Proposal)
        initiator = self.agents[0]
        proposal_msg = self.call_agent_func(initiator, "propose_movie", kwargs={"cur_time": cur_time})
        proposed_movie = proposal_msg.content
        logger.info(f"Group [{self.name}] selected movie: {len(proposed_movie)}")
        logger.info(f"proposed_movie: {proposed_movie}")
        if not proposed_movie or not proposed_movie.get("movie_id"):
            logger.warning(f"Group [{self.name}] failed to select a movie or movie_id is missing.")
            return Msg(self.name, role="assistant", content=[])

        # 2. 讨论 (Discussion) - 修正为串行逻辑
        discussion_context = ""
        icebreaker = self.agents[0]
        initial_thoughts_msg = self.call_agent_func(
            icebreaker, "generate_initial_thoughts",
            kwargs={"movie_info": proposed_movie}
        )
        discussion_context += f"{icebreaker.name}'s opening thought: {initial_thoughts_msg.content}\n\n"
        
        for member in self.agents[1:]:
             contribution_msg = self.call_agent_func(
                 member, "participate_in_discussion",
                 kwargs={"movie_info": proposed_movie, "discussion_context": discussion_context}
             )
             discussion_context += f"{member.name}'s reply: {contribution_msg.content}\n\n"
        logger.info(f"Group [{self.name}] discussion context length: {len(discussion_context)}")
        # logger.info(f"content: {discussion_context}")
        logger.info(f"Group [{self.name}] finished discussion.")

        # 3. 舆论洞察 (Public Opinion Insight)
        logger.info(f"Group [{self.name}] gathering public reviews for movie ID {proposed_movie['movie_id']}...")
        try:
            reviews_msg = self.call_manager_agent_func(
                "call_tool_func",
                kwargs={
                    "function_call_msg": [{
                        "tool": "GetReviewsForMovie",
                        "tool_input": {"movie_id": proposed_movie['movie_id']}
                    }]
                }
            )
            logger.info(f"reviews_msg: {reviews_msg}")
            logger.info(f"reviews_msg.content: {reviews_msg.content}")
            public_reviews_info = reviews_msg.content[0][1]
            existing_reviews_context = public_reviews_info['result']
            logger.info(f"Group [{self.name}] got public reviews:{public_reviews_info}.\nRaw review msg: {reviews_msg.content}")
        except Exception as e:
            logger.warning(f"Could not parse reviews for group [{self.name}]: {e}")
            existing_reviews_context = ""

        # 4. 生成最终评论 (Concurrent Review Generation)
        review_placeholders = [
            self.call_agent_func(
                member, "rate_movie_process",
                kwargs={
                    "cur_time": cur_time,
                    "discussion_context": discussion_context,
                    "existing_reviews_context": existing_reviews_context,
                    "pre_selected_movie": proposed_movie
                }
            ) for member in self.agents
        ]

        all_final_ratings = []
        for msg_placeholder in review_placeholders:
            result_content = msg_placeholder.content
            if result_content:
                all_final_ratings.extend(result_content)
        
        logger.info(f"Group [{self.name}] collected {len(all_final_ratings)} ratings.")
        logger.info(f"all_final_ratings: {all_final_ratings}")
        return Msg(self.name, role="assistant", content=all_final_ratings)


    def reply(self, message: Msg) -> Msg:
        func_name = message.get("func","")
        kwargs = message.get("kwargs",{})
        func = getattr(self,func_name)
        res = func(**kwargs)
        assert isinstance(res,Msg)

        return res