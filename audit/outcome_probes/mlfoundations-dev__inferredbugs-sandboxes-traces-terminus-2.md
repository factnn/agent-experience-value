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
  - name: date
    dtype: string
  - name: task
    dtype: 'null'
  - name: episode
    dtype: string
  - name: run_id
    dtype: string
  - name: trial_name
    dtype: string
  splits:
  - name: train
    num_bytes: 568134132
    num_examples: 9992
  download_size: 162597548
  dataset_size: 568134132
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
