# -- 新增代码 --
from . import movie_prompt_registry, movie_prompt_default
from Emulate.prompt.base import BaseChatPromptTemplate

@movie_prompt_registry.register("propose_movie")
class ProposeMoviePromptTemplate(BaseChatPromptTemplate):
    def __init__(self, **kwargs):
        template = kwargs.pop("template", movie_prompt_default.get("propose_movie", ""))
        input_variables = kwargs.pop("input_variables", [
            "group_genres", "movie_description", "agent_scratchpad"
        ])
        super().__init__(template=template, input_variables=input_variables, **kwargs)

@movie_prompt_registry.register("discuss_movie")
class DiscussMoviePromptTemplate(BaseChatPromptTemplate):
    def __init__(self, **kwargs):
        template = kwargs.pop("template", movie_prompt_default.get("discuss_movie", ""))
        input_variables = kwargs.pop("input_variables", [
            "role_description", "movie_title", "discussion_context", "agent_scratchpad"
        ])
        super().__init__(template=template, input_variables=input_variables, **kwargs)
# -- 新增结束 --