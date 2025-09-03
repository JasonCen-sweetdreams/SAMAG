import yaml
from .. import MODEL
control_prompt_default = yaml.safe_load(open(f"Emulate/prompt/control/{MODEL}.yaml"))

from Emulate.registry import Registry
control_prompt_registry = Registry(name="ControlPromptRegistry")

from .control import *