# HF Autoresearch: Create Experiment

Scaffolds and runs HuggingFace fine-tuning experiments from natural language requests like "fine-tune X on Y".

## Trigger

Use when the user asks to:
- Fine-tune a model on a dataset
- Run autoresearch / hyperparameter search
- Train a model with automatic tuning
- Start an experiment loop

## Workflow

### 1. Parse the Request

Extract from the user's request:
- **Model name** (HuggingFace model ID, e.g. `bert-base-uncased`, `meta-llama/Llama-2-7b-hf`)
- **Dataset name** (HuggingFace dataset ID, e.g. `imdb`, `glue/sst2`)
- **Task type** (if not specified, infer from model/dataset)

### 2. Inspect Model and Dataset

Before scaffolding, **always** inspect the model card and dataset card:

```
fetch_content: https://huggingface.co/{model_name}
fetch_content: https://huggingface.co/datasets/{dataset_name}
```

From the model card, determine:
- Architecture (encoder-only, decoder-only, encoder-decoder)
- Pre-training task (MLM, CLM, seq2seq)
- Appropriate task type for fine-tuning

From the dataset card, determine:
- Column names (text column, label column, second text column for pairs)
- Number of labels (for classification)
- Dataset config/subset name if needed
- Dataset size (affects batch size / epoch choices)
- Primary evaluation metric

### 3. Scaffold the Experiment

Call `scaffold_experiment` with the gathered information:

```
scaffold_experiment:
  modelName: "{model}"
  datasetName: "{dataset}"
  datasetConfig: "{config if needed}"
  task: "{task type}"
  metricName: "{primary metric}"
  metricDirection: "{higher or lower}"
  textColumn: "{text column name}"
  textColumn2: "{second text column, for NLI/paraphrase}"
  labelColumn: "{label column name}"
  numLabels: {number if classification}
  maxLength: {appropriate for model}
  useLora: {true for models >1B params}
  loraR: {16 default, increase for more capacity}
  useQlora: {true for very large models on CUDA}
```

**Task type mapping:**
| Model Type | Dataset Type | Task |
|---|---|---|
| BERT, RoBERTa, DeBERTa | Sentiment, topic, NLI | `text-classification` |
| BERT, RoBERTa | NER, POS tagging | `token-classification` |
| BERT, RoBERTa | SQuAD-style | `question-answering` |
| T5, BART | Summarization | `summarization` |
| T5, BART | Translation | `translation` |
| GPT-2, LLaMA, Mistral | Text generation | `causal-lm` |

**Default metrics:**
| Task | Default Metric | Direction |
|---|---|---|
| text-classification | eval_accuracy | higher |
| token-classification | eval_f1 | higher |
| question-answering | eval_loss | lower |
| summarization | eval_rougeL | higher |
| causal-lm | eval_loss | lower |

**Sentence pair datasets** (NLI, paraphrase, STS):
- Set `textColumn` to first sentence column (e.g. `"premise"`, `"sentence1"`)
- Set `textColumn2` to second sentence column (e.g. `"hypothesis"`, `"sentence2"`)

### 4. (Optional) Customize train.py

If the model/dataset needs special handling, edit the generated `train_{experiment_id}.py`:
- Custom preprocessing (e.g., formatting prompts for instruction tuning)
- Custom model head or loss function
- Special tokenization (e.g., for multi-turn dialogue)
- Dataset filtering or sampling

### 5. Run Baseline Experiment

Start with conservative defaults:

```
run_experiment:
  experimentId: "{experiment_id}"
  hyperparams:
    lr: 2e-5
    batch_size: 16
    epochs: 3
    warmup_ratio: 0.1
    weight_decay: 0.01
  description: "Baseline run with default hyperparams"
```

**Baseline hyperparams by model size:**
| Model Size | LR | Batch Size | Epochs | LoRA? |
|---|---|---|---|---|
| < 200M (BERT, DistilBERT) | 2e-5 | 16-32 | 3-5 | No (full fine-tune) |
| 200M-1B (RoBERTa-large, DeBERTa) | 1e-5 | 8-16 | 3 | Optional |
| 1B-7B (LLaMA, Mistral) | 2e-4 | 4-8 | 1-2 | **Yes (LoRA)** |
| > 7B | 1e-4 | 2-4 | 1 | **Yes (QLoRA)** |

For LoRA experiments, use higher LR (1e-4 to 3e-4) than full fine-tuning.

**Quick smoke test first:** Use `max_train_samples: 500` and `max_eval_samples: 200` to verify the pipeline works before committing to a full run.

### 6. Log the Result

**IMPORTANT:** After every `run_experiment`, you MUST call `log_experiment` to record the verdict.
Runs start with status "running" and must be explicitly logged.

```
log_experiment:
  runId: "{run_id}"
  status: "keep"
  notes: "Baseline: accuracy=0.89, loss=0.32. Good starting point."
```

- **keep**: Result is promising, use as reference for future runs
- **discard**: Completed but not useful (worse metrics, wrong direction)
- **crash**: Failed (OOM, timeout, errors)

### 7. Hyperparameter Loop

Now iterate. **Reason about what to try next** based on results:

**Strategy guidelines:**
1. **First 3 runs:** Explore learning rate (10x range: 1e-5, 5e-5, 1e-4)
2. **Next 2 runs:** Explore batch size (halve and double from baseline)
3. **Then:** Fine-tune best combo — try warmup, weight decay, epochs
4. **If overfitting:** Increase weight decay, reduce epochs, add dropout, try `early_stopping_patience: 3`
5. **If underfitting:** Increase LR, increase epochs, try larger model or increase `lora_r`
6. **If OOM:** Reduce batch size, add gradient accumulation, reduce max_length, use LoRA/QLoRA

**After each run:**
1. Compare metrics to previous best (delta shown in output)
2. Log keep/discard with reasoning
3. Decide what to try next based on the trend
4. Stop when: metric plateaus for 3+ runs, or budget exhausted

**Example reasoning:**
```
Run 1 (baseline): accuracy=0.89, lr=2e-5, bs=16
  → log_experiment: keep "baseline, decent starting point"
Run 2: accuracy=0.91, lr=5e-5, bs=16
  → log_experiment: keep "higher LR helped +2.2%, try higher"
Run 3: accuracy=0.88, lr=1e-4, bs=16
  → log_experiment: discard "too high, overshot optimal LR"
Run 4: accuracy=0.92, lr=5e-5, bs=32
  → log_experiment: keep "larger batch with best LR, new best +1.1%"
Run 5: accuracy=0.92, lr=5e-5, bs=32, warmup=0.2
  → log_experiment: discard "no improvement, plateau reached"
```

### 8. Report Results

After the loop, summarize:
- Best hyperparameters found
- Best metric achieved
- Improvement over baseline (%)
- Total runs and time spent
- Recommendations for further improvement

## Storage Layout

All heavy data stored on external drive:
```
/Volumes/Expansion/hf-autoresearch/
├── models/      # Cached HF model weights
├── datasets/    # Cached HF datasets
├── runs/        # Training checkpoints and logs
│   └── {run_id}/
│       ├── train.log      # Full training output
│       ├── run_info.json  # Hyperparams + metrics
│       └── checkpoint-*/  # Model checkpoints
├── results/     # (reserved for future use)
└── cache/
    └── uv/      # uv package cache
```

Project-local state:
```
{project}/
├── hf-autoresearch.jsonl     # Experiment configs and run records
└── train_{experiment_id}.py  # Generated training scripts
```

## Runtime

The extension uses `uv run --script` to execute train.py. The PEP 723 inline metadata
in train.py declares all Python dependencies (torch, transformers, datasets, etc.).
On first run, `uv` automatically creates an isolated environment and installs deps.
Subsequent runs reuse the cached environment.

If `uv` is not available, falls back to `python3` (requires manual pip install).

## Tips

- **Check disk space** on /Volumes/Expansion before starting large experiments
- **Use Ctrl+Shift+X** to open the autoresearch dashboard at any time
- **Use /autoresearch** command for the full dashboard view
- The agent should explain its reasoning for each hyperparameter choice
- For large models (>7B), consider using LoRA/QLoRA via the unsloth or axolotl skills instead
- The system prompt automatically includes experiment state, so the agent always knows the current context
