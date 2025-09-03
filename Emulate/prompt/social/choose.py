from Emulate.message import Message
from Emulate.prompt.social import social_prompt_registry,social_prompt_default

from Emulate.prompt.base import BaseChatPromptTemplate  

# Set up a prompt template
@social_prompt_registry.register("choose_topic")
class ChooseTopicPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             social_prompt_default.get("choose_template",""))

        input_variables = kwargs.pop("input_variables",
                    [ 
                     "role_description",
                     "memory",
                     "twitter_topic",
                     "agent_scratchpad"])
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
