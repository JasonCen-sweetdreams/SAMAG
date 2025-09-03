import yaml
from .. import MODEL
article_prompt_default = yaml.safe_load(open(f"Emulate/prompt/article/{MODEL}.yaml"))

from Emulate.registry import Registry
article_prompt_registry = Registry(name="ArticlePromptRegistry")

from .group_discuss import *
from .get_author import *
from .choose import *
from .write_article import *
