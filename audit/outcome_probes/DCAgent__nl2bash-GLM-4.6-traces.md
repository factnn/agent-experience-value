---
dataset_info:
  features:
  - name: conversations
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: agent
    dtype: string
  - name: model
    dtype: string
  - name: model_provider
    dtype: string
  - name: date
    dtype: string
  - name: task
    dtype: string
  - name: episode
    dtype: string
  - name: run_id
    dtype: string
  - name: trial_name
    dtype: string
  splits:
  - name: train
    num_bytes: 80183428
    num_examples: 6503
  download_size: 18786097
  dataset_size: 80183428
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
