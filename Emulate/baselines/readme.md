# Micro-scale Evaluation 

## train dataset
```bash
python Emulate/baselines/get_expansion_dataset.py
```
or use default: `Emulate/baselines/baseline_checkpoints/llmcitationciteseer.pkl`



## generate sampled samag results
```bash
python Emulate/baselines/get_llmgen_graphs.py
```

## evaluate
```bash
python Emulate/baselines/eval_pred_graphs.py
```

And the results will be stored in `./graph_gen_df`

# Macro-scale Evaluation
```bash
python evaluate/article/main.py

python evaluate/social/main.py

python evaluate/movie/main.py
```
<!-- # macro evaluation
result checkpoints
"Emulate/tasks/citeseer/configs/fast_vllm"
"Emulate/tasks/movielens/configs/test_movie_up"
"Emulate/tasks/tweets/configs/llama_test_1e6" -->