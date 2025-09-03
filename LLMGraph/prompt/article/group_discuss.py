from LLMGraph.message import Message
from LLMGraph.prompt.article import article_prompt_default,article_prompt_registry

from LLMGraph.prompt.base import BaseChatPromptTemplate
    
    
@article_prompt_registry.register("group_discuss")
class GroupDiscussPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             article_prompt_default.get("group_discuss_template",""))

        input_variables = kwargs.pop("input_variables",
                    ["character_1",
                     "character_2",
                     "past_context",
                     "cur_context",
                     "research_content",
                     "agent_scratchpad",
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    

@article_prompt_registry.register("choose_researcher")
class ChooseResearcherPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             article_prompt_default.get("choose_researcher_template",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description",
                     "research_topic",
                     "past_context",
                     "researcher",
                     "agent_scratchpad",
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
        
@article_prompt_registry.register("get_idea")
class GetIdeaPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             article_prompt_default.get("get_idea",""))

        input_variables = kwargs.pop("input_variables",
                    ["role_description",
                     "research_idea",
                     "past_context",
                     "researcher",
                     "agent_scratchpad",
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
        

@article_prompt_registry.register("proposal_initialization")
class ProposalInitializationPromptTemplate(BaseChatPromptTemplate):
    def __init__(self, **kwargs):
        template = kwargs.pop(
            "template",
            article_prompt_default.get("proposal_initialization_template", "")
        )
        input_variables = kwargs.pop(
            "input_variables",
            [
                "role_description",
                "past_context",
                "topic",
                "interested_topics",
                "agent_scratchpad",
                "searched_info"
            ]
        )
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)

@article_prompt_registry.register("suggestion_generation")
class SuggestionGenerationPromptTemplate(BaseChatPromptTemplate):
    def __init__(self, **kwargs):
        template = kwargs.pop(
            "template",
            article_prompt_default.get("suggestion_generation_template", "")
        )
        input_variables = kwargs.pop(
            "input_variables",
            [
                "role_description",
                "past_context",
                "proposals",
                "agent_scratchpad",
                "searched_info"
            ]
        )
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)

@article_prompt_registry.register("proposal_generation")
class ProposalGenerationPromptTemplate(BaseChatPromptTemplate):
    def __init__(self, **kwargs):
        template = kwargs.pop(
            "template",
            article_prompt_default.get("proposal_generation_template", "")
        )
        input_variables = kwargs.pop(
            "input_variables",
            [
                "role_description",
                "past_context",
                "proposals",
                "agent_scratchpad",
            ]
        )
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
