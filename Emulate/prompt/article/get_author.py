from Emulate.message import Message
from Emulate.prompt.article import article_prompt_default,article_prompt_registry

from Emulate.prompt.base import BaseChatPromptTemplate
    
    
@article_prompt_registry.register("get_author")
class GetAuthorPromptTemplate(BaseChatPromptTemplate):
    
    def __init__(self,**kwargs):
        template = kwargs.pop("template",
                             article_prompt_default.get("get_author",""))

        input_variables = kwargs.pop("input_variables",
                    ["expertises_list",
                     "author_num",
                     ])
        
        super().__init__(template=template,
                         input_variables=input_variables,
                         **kwargs)
    
