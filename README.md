# Sensitive Prompt Study Artifacts

This repository contains code and data artifacts for two research-question tracks centered on sensitive information in LLM prompts:

- `RQ1`: labeled datasets and annotation artifacts.
- `RQ2`: prompt-sensitivity classification baselines, training, and evaluation.

The repository is organized so each `RQ*` directory can be used independently.

## Repository Layout

```text
.
├── RQ1/   # datasets and annotation files
├── RQ2/   # classification experiments and statistical analysis
```

## RQ1: Datasets and Annotation Artifacts

`RQ1` contains the manually reviewed and source datasets used for prompt-sensitivity analysis.

Files:

- `codechat_manually_judged_prompts.json`: manual labels for the CodeChat dataset after regex filtering.
- `swechat_manually_judged_prompts.json`: manual labels for the SWE-Chat dataset (agentic coding session prompts) after regex filtering.
- `edevgpt_prompt_sensitive_only.json`: manual labels for the sensitive EDevGPT subset kept in the artifact.
- `edev_gpt_unique_prompts_dataset.json`: source EDevGPT prompt corpus.
- `author1.json` and `author2.json`: 500-example annotation files for inter-rater agreement calculations.

See [RQ1/README.md] for the file-level notes already included with the artifact.

## RQ2: Prompt-Sensitivity Classification

`RQ2` contains the training and evaluation pipeline for binary prompt-sensitivity classification.

Included data:

- `data/phase1_dataset.json`
- `data/phase2_dataset.json`
- `data/external_dataset.json`

Included scripts:

- `baselines.py`: majority-class, regex/keyword, TF-IDF word, and TF-IDF character baselines.
- `run_single_experiment.py`: train one encoder/pooling configuration.
- `run_all_experiments.py`: launch all model/pooling combinations.
- `evaluate_external.py`: evaluate trained checkpoints on the external dataset.
- `statistical_tests.py`: bootstrap confidence intervals and significance testing across phases.
- `threshold_analysis.py`: operating-point and deployment-threshold summary.

Models evaluated by `run_all_experiments.py`:

- `bert-base-uncased`
- `roberta-base`
- `microsoft/deberta-base`
- `microsoft/codebert-base`
- `microsoft/unixcoder-base`
- `distilbert-base-uncased`

Each model is run with `cls`, `mean`, and `max` pooling.

### RQ2 Setup

```bash
cd RQ2
pip install -r requirements.txt
```

### RQ2 Reproduction

Run baselines:

```bash
python3 baselines.py \
  --dataset data/phase1_dataset.json \
  --eval-data data/external_dataset.json \
  --output results/baselines_results.json
```

Run all Phase 1 experiments and evaluate externally:

```bash
python3 run_all_experiments.py \
  --dataset data/phase1_dataset.json \
  --output-dir results/phase1_experiment_outputs

python3 evaluate_external.py \
  --experiment-dir results/phase1_experiment_outputs \
  --eval-data data/external_dataset.json \
  --output results/phase1_external_eval_results.json
```

Run all Phase 2 experiments and evaluate externally:

```bash
python3 run_all_experiments.py \
  --dataset data/phase2_dataset.json \
  --output-dir results/phase2_experiment_outputs

python3 evaluate_external.py \
  --experiment-dir results/phase2_experiment_outputs \
  --eval-data data/external_dataset.json \
  --output results/phase2_external_eval_results.json
```

Run statistical comparison and threshold analysis:

```bash
python3 statistical_tests.py \
  --phase1-external results/phase1_external_eval_results.json \
  --phase2-external results/phase2_external_eval_results.json \
  --output results/statistical_results.json

python3 threshold_analysis.py \
  --phase1-external results/phase1_external_eval_results.json \
  --phase2-external results/phase2_external_eval_results.json \
  --output results/threshold_analysis_results.json
```

`run_all_experiments.py` schedules two parallel workers and maps them onto GPU IDs `0` and `1` through `CUDA_VISIBLE_DEVICES`. If you do not have a two-GPU setup, use `run_single_experiment.py` directly or adapt the launcher.

See [RQ2/README.md] for the original reproduction outline.


## Notes

- There is no single top-level dependency file for the whole repository; install dependencies per `RQ2`.
- `RQ1` is data-only.
- Many scripts write outputs into `results/` directories that are not committed in this repository.
