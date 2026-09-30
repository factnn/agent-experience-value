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
    num_bytes: 241543913
    num_examples: 9956
  download_size: 78190704
  dataset_size: 241543913
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
