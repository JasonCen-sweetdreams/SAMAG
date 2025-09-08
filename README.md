# SAMAG: Structure-Aware Multi-Agent Graph Generation with Large Language Models

## 0. Environment Setup
Create a virtual environment for SAMAG
```bash
conda create --name <env_name> python=3.9
conda activate <env_name>
```
Pip install agentscope[distributed] v0.0.4 (from https://github.com/modelscope/agentscope/)
```bash
git clone https://github.com/modelscope/agentscope/
git reset --hard 1c993f9
# From source
pip install -e .[distribute]
```
Then install the required packages:
```bash
pip install -i "requirements.txt"
```

## 1. Run SAMAG Emulation
Firstly, please set up your API keys in `Emulate\llms\default_model_configs.json`:
```json
[
    // GPT API
    {
        "model_type": "openai_chat",
        "config_name": "gpt-3.5-turbo-0125",
        "model_name": "gpt-3.5-turbo-0125",
        "api_key": "sk-*",
        "generate_args": {
            "max_tokens": 2000,
            "temperature": 0.8
        }
},
    // VLLM Server API
 {
        "config_name": "llama3-70B",
        "model_type": "openai_chat",
        "model_name": "llama3-70B",
        "api_key": "vllm_server_api",
        "client_args": {
            "base_url": "vllm_server_url"
        },
        "generate_args": {
            "temperature": 0.9,
            "max_tokens": 2000
        }
    }

]
```
Then set up the work directory:
```bash
export PYTHONPATH=./
```

- To start building citation network in LLMGraph, you should first specify the dir of data and the config name, and then simply run by
    ```bash
    python start_launchers.py
    python main.py --task citeseer --config "small" --build
    ```

- To start building social network in Emulate, you should first specify the dir of data and the config name, and then simply run by
    ```bash
    python start_launchers.py
    python main.py --task tweets --config "small" --build
    ```

- To start building movie rating network in Emulate, you should first specify the dir of data and the config name, and then simply run by
    ```bash
    python start_launchers.py
    python main.py --task movielens --config "small" --build
    ```

## 2. Evaluation
### Graph Structure Alignment
Please refer to [README For Graph Structure Alignment](./Emulate/baselines/readme.md)

### Textual Feature Alignment
> Coming soon...