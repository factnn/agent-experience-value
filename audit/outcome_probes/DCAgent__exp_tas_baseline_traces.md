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
  - name: result
    dtype: string
  - name: trace_source
    dtype: string
  splits:
  - name: train
    num_bytes: 368202111
    num_examples: 10393
  download_size: 121918223
  dataset_size: 368202111
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
