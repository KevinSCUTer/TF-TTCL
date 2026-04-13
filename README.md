<h1 align="center">
     <br>Training-Free Test Time Constrastive Learning for Large Language Models
</h1>


<h4 align="center"></a>


>[Kaiwen Zheng](https://scholar.google.com/citations?user=mHTk234AAAAJ&hl=en), [Kai Zhou](https://scholar.google.com/citations?hl=en&user=58UyQ9cAAAAJ&hl=en), [Jinwu Hu](https://scholar.google.com/citations?user=XmqjPi0AAAAJ&hl=en), Te Gu, Mingkai Peng, [Fei Liu](https://scholar.google.com/citations?hl=en&user=gC-YMYgAAAAJ)
><sub>South China University of Technology, Pazhou Laboratory</sub>

# TF-TTCL

Training-Free Test-Time Contrastive Learning (TF-TTCL) is a training-free framework for improving frozen or API-accessed LLMs during inference. It learns from test-time experience by contrasting better and worse reasoning trajectories, distilling them into reusable rules, and retrieving those rules for later questions.

Accepted by Findings of ACL 2026.

## Repository Layout

```text
core/
├── main.py                 # Experiment runner
├── config/                 # Config templates
└── source/                 # Actors, retrieval, selection, summary, prompts

data/
├── MATH/                   # Processed math datasets
└── OPEN/                   # Processed open-domain datasets

evaluate/                   # Offline evaluation scripts
LICENSE
requirements.txt
```

## Installation

We recommend using **[uv](https://github.com/astral-sh/uv)** and Python 3.12 for lightning-fast environment setup.

```bash
git clone https://github.com/KevinSCUTer/TF-TTCL.git
cd TF-TTCL
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Datasets

This repository includes processed datasets for:

- Math: `gsm8k`, `math500`, `aime24`, `minerva`
- Open-domain QA: `agriculture`, `geography`, `medicine`, `wealth`

Current processed files live at:

- `data/MATH/{dataset_name}.json`
- `data/OPEN/{dataset_name}.json`

Reference/raw dataset files are kept under `data/MATH/reference/`.

## Quick Start

TF-TTCL expects:

- an OpenAI-compatible chat completion endpoint
- an embedding endpoint when `rag.enabled: true`

Start from the config template:

```bash
cp core/config/config.yaml core/config/local.yaml
```

Then edit `core/config/local.yaml`.

> Note: mode and domain have only 5 kinds of composition:
> mode "close" with domain "math"
> mode "open" with domain "agriculture"/"medicine"/"geography"/"wealth"

```yaml
experiment:
  data_path: "/path/to/TF-TTCL/data/MATH/gsm8k.json"
  limit: 10
  mode: "close"
  domain: "math"

prompt_file_path:
  teacher: "/path/to/TF-TTCL/core/source/prompts/close/role/math/teacher/gsm.md"
  ta: "/path/to/TF-TTCL/core/source/prompts/close/role/math/ta/simple_rephrase.md"
  student: "/path/to/TF-TTCL/core/source/prompts/close/role/math/student/gsm.md"
  positive_batch: "/path/to/TF-TTCL/core/source/prompts/close/rule/positive/batch_extract.md"
  negative_batch: "/path/to/TF-TTCL/core/source/prompts/close/rule/negative/batch_extract.md"

llm:
  api_url: "your-openai-compatible-url/v1"
  api_key: "your_api_key"
  model_name: "your_model_name"
  max_tokens: 8192 # For AIME24, set to 16384 or larger to prevent truncation. For other datasets, 8192 is sufficient.
  max_context_tokens: 32768

rag:
  enabled: true
  embedding_api_url: "your-embedding-url/v1"
  embedding_api_key: "your_embedding_api_key"
  embedding_model: "Qwen/Qwen3-Embedding-0.6B"

rules:
  max_pos_rules: 3
  max_neg_rules: 3

student:
  number: 4
```

Run an experiment with:

```bash
python core/main.py --config core/config/local.yaml
```

Useful overrides:

```bash
python core/main.py --config core/config/local.yaml --limit 50
python core/main.py --config core/config/local.yaml --override rules.max_pos_rules=5 rules.max_neg_rules=5
```

For quick customization without editing the config file, use the override script:

```bash
bash core/run_override.sh
```

This is equivalent to:

```bash
python core/main.py --config absolute/path/to/your_config.yaml \
  --override experiment.output_dir=exp_res/your_experiment_name \
           student.number=4 \
           batch.size=1 \
           rules.max_pos_rules=30 \
           rules.max_neg_rules=30
```

## Outputs

Results are written to:

```text
core/exp_res/<dataset>_<mode>_<yyyymmdd>_<hhmmss>/
```

Typical outputs include:

- `experiment.log`
- `output.jsonl`
- `rules.json`
- `summary.json`
- `question_variants.jsonl` when `debug.save_variants: true`

## Evaluation

Offline evaluation scripts are in `evaluate/`.

> Note: the evaluation scripts are lightweight research utilities. Before running them, update the input paths inside the scripts so they point to your generated results and local dataset/reference files.

### Math tasks

```python
# Please modify to absolute paths if you want to use the ground truth labels for GSM8K, MATH-500, Minerva, AIME24 datasets. 
GSM8K_PARQUET_PATH = "path/to/data/MATH/reference/gsm8k/main/test-00000-of-00001.parquet"
MATH500_JSONL_PATH = "path/to/data/MATH/reference/MATH-500/test.jsonl"
MINERVA_JSONL_PATH = "path/to/data/MATH/reference/minerva/test.jsonl"
AIME_JSONL_PATH = "path/to/data/MATH/reference/aime24/aime.jsonl"
```

Then run:

```bash
python evaluate/accuracy/eval_accuracy.py
```

### Open-domain tasks

Create a new conda environment and install the extra evaluation dependencies if needed:

```bash
pip install rouge_score nltk bert_score git+https://github.com/google-research/bleurt.git
```

Then run:

```bash
python evaluate/similarity/eval_similarity.py
```

## Citation

Thanks for the open-source code of [TLM](https://github.com/Fhujinwu/TLM) and [Verbalized Sampling](https://www.verbalized-sampling.com/)

If you find our work interesting and meaningful, welcome to give a 🌟 to our repo and cite our paper.

```bib
@inproceedings{tfttcl,
    title={Training-Free Test-time Contrastive Learning for Large Language Models},
    author={Zheng, Kaiwen and Zhou, Kai and Hu, Jinwu and Gu, Te and Peng, Mingkai and Liu, Fei},
    booktitle={Findings of the Association for Computational Linguistics: ACL 2026}
}
```