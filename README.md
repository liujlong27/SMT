# Beyond the Prompt: Jailbreaking Function-Calling LLMs via Simulated Moderation Traces

## Introduction

This repository contains the official implementation of our paper:

> **Beyond the Prompt: Jailbreaking Function-Calling LLMs via Simulated Moderation Traces**

If you find this repository useful for your research, please consider citing our paper. Questions, bug reports, and reproducibility issues are welcome through GitHub Issues.

## Overview

![framework](./figures/pipline.jpg)

## Repository Structure

```text
.
├── install.sh                      # Installation script for setting up the environment and dependencies
├── main.py                         # Main evaluation entry point
├── requirements.txt                # Python dependencies
├── data/
│   ├── SafeBench-Tiny.txt          # Default benchmark
│   └── JailbreakBench.txt          # Optional benchmark
├── figures/
│   ├── pipline.png                 # Framework overview
│   ├── main_res1.png               # Main result figure
│   └── main_res2.png               # Main result figure
├── utils/
│   ├── config.example.json         # Example API configuration
│   ├── eval_h_cot.py               # External evaluation utility
│   ├── fake_tool_call_noharmful.json
│   ├── moderation_generator.py
│   └── moderation_validator.py
└── output/                         # Checkpoints and evaluation results
```

## Getting Started

### 1. Create the Environment

We provide an installation script to automatically set up the Python environment and install all required dependencies.

```bash
chmod +x install.sh
./install.sh
```

After the installation is complete, activate the virtual environment:

```bash
source .venv/bin/activate
```

Alternatively, you can manually create the environment using Conda:

```bash
conda create -n SMT python=3.12
conda activate SMT
pip install -r requirements.txt
```

### 2. Configure API Access

Create `utils/config.json` from the example file:

```bash
cp utils/config.example.json utils/config.json
```

A minimal configuration file should follow this format:

```json
{
  "llm_api": {
    "base_url": "https://your-api-endpoint/v1",
    "api_key": "YOUR_API_KEY"
  },
  "local": {
    "base_url": "http://localhost:1234/v1",
    "api_key": "YOUR_LOCAL_API_KEY"
  }
}
```

The two entries correspond to:

* `llm_api`: the default OpenAI-compatible API endpoint for commercial models.
* `local`: a local OpenAI-compatible inference endpoint, used for selected local models.

### 3. Select a Benchmark and Target Model

Open `main.py` and modify the experiment configuration near the beginning of the file:

```python
DATA_FILE = "SafeBench-Tiny.txt"
# DATA_FILE = "JailbreakBench.txt"

TARGET_MODELS = [
    "gpt-4o-2024-11-20",
    # "gpt-5.4-2026-03-05",
    # "qwen3-max-2026-01-23",
    # "gemini-3-flash-preview",
    # "deepseek-v4-flash",
    # "claude-sonnet-4-5-20250929", 
]
```

Use model identifiers that are valid for the API endpoint configured in `utils/config.json`.

### 4. Configure Evaluation Parameters

The default parameters are defined in `main.py`:

```python
MAX_THREADS = 1
MAX_ROUNDS = 3
TURNS_NUM = 3
RESET = False
```

Their meanings are:

| Parameter     | Description                                                 |
| ------------- | ----------------------------------------------------------- |
| `MAX_THREADS` | Maximum concurrent workers per target model                 |
| `MAX_ROUNDS`  | Maximum outer retry rounds                                  |
| `TURNS_NUM`   | Maximum multi-turn interaction attempts per sample          |
| `RESET`       | Whether to discard an existing checkpoint before evaluation |

Set:

```python
RESET = True
```

to start a fresh run for the selected model. Leave it as `False` to resume from an existing checkpoint.

### 5. Run the Evaluation

```bash
python main.py
```

For each target model, the script automatically:

1. Loads the selected benchmark from `data/`.
2. Runs SMT against all pending samples.
3. Saves intermediate progress after each sample.
4. Resumes interrupted experiments from saved checkpoints.
5. Reports aggregate metrics after evaluation.

## Output Format

Results are saved under:

```text
output/attack_checkpoint_<dataset>_<model>.json
```

For example:

```text
output/attack_checkpoint_SafeBench-Tiny_gpt-4o-2024-11-20.json
```

Each benchmark item is stored with fields similar to:

```json
{
  "target_question": {
    "success": true,
    "queries": 3,
    "payload": "...",
    "score": 60
  }
}
```

The script reports the following metrics:

| Metric        | Description                                                                             |
| ------------- | --------------------------------------------------------------------------------------- |
| **ASR**       | Attack Success Rate: percentage of samples whose score reaches the configured threshold |
| **Avg.Q**     | Average number of API queries used per sample                                           |
| **Avg.Score** | Average external evaluation score across all samples                                    |

## Reproducibility Notes

Commercial LLM APIs are continuously updated. Even when using the same model alias, results may vary due to provider-side model updates, routing behavior, safety-policy changes, decoding settings, or transient API failures.

To improve reproducibility:

* Prefer dated model snapshots whenever available.
* Keep the benchmark files unchanged.
* Record the provider, endpoint, model ID, date, and parameters for each run.
* Use the same threshold and query budget reported in the paper.

All experiments reported in this repository were conducted in June 2026. Because the simulated validation traces include randomized elements, model outputs are inherently stochastic, and both model capabilities and external detection mechanisms continue to evolve, exact interaction histories and reproduced results may differ from those reported here even under the same high-level configuration.

## Main Results

![main\_results](./figures/main_res1.jpg)

![main\_results](./figures/main_res2.jpg)

Our evaluation compares SMT with representative jailbreak baselines across multiple commercial function-calling LLMs and two safety benchmarks.

Please refer to the paper for full experimental settings, baseline implementations, statistical details, and defense analysis.

## Ethical Considerations

This project studies failure modes in safety-aligned, function-calling LLM systems. The code and associated materials are intended to support:

* Authorized red-team evaluations
* Safety benchmarking
* Research on tool-use and agent security
* Development of context-aware guardrails
* Reproducibility of published research

Users are responsible for complying with applicable laws, institutional review requirements, API-provider terms, and responsible-disclosure practices.

We strongly discourage deploying this framework against production systems without authorization. Researchers should avoid exposing raw harmful outputs, API credentials, personally identifiable information, or unrestricted attack artifacts in public repositories.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{liu2027smt,
    title={Beyond the Prompt: Jailbreaking Function-Calling LLMs via Simulated Moderation Traces},
    author={Junlong Liu and Haobo Wang and Weiqi Luo and Xiaojun Jia},
    booktitle={{IEEE} Symposium on Security and Privacy},
    year={2027}
}
```
