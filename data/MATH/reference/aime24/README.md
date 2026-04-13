---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: problem
    dtype: string
  - name: solution
    dtype: string
  - name: url
    dtype: string
  splits:
  - name: test
    num_bytes: 13290
    num_examples: 30
  download_size: 11183
  dataset_size: 13290
configs:
- config_name: default
  data_files:
  - split: test
    path: test-*
license: apache-2.0
---

# AIME 24

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0) 

### American Invitational Mathematics Examination (AIME) 2024

## Citation
If you use the AIME24 dataset in your research, please consider citing it as follows:

```
@misc{aime24,
      title={American Invitational Mathematics Examination (AIME) 2024}, 
      author={Zhang, Yifan and Math-AI, Team},
      year={2024},
}
```
