# PyTorch context classifier

A local supervised classifier that reads `query_patterns` as inputs and `context` as labels. No external LLM APIs, pretrained model, or model download is used. Package installation needs access to Python packages; training and prediction then run locally on CPU.

## Your supplied data

`user_sample.json` preserves your 12 patterns and fixes the missing comma before `context`. It contains only `case_details`. A conventional multiclass classifier needs at least two distinct contexts. With one softmax output, every input receives probability 1, including unrelated text. Training deliberately refuses that dataset alone.

Add your real other contexts with representative patterns. If you only need to detect `case_details`, supply varied negative examples under a second label such as `other`. An `other` class will not cover every possible unknown input.

`demo_train.json` contains your sample plus additional case-details wording, and two invented example contexts: `installment_details` and `payment_history`. These are only a runnable demonstration, not assumed business requirements. `demo_test.json` is a separate nine-query illustration.

## Installation

Use Python 3.11 or later. In the extracted project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For Windows activation use `.venv\Scripts\activate`. This implementation runs on CPU. You can install an appropriate CPU-only PyTorch distribution using the official PyTorch installation selector if you do not want GPU dependencies.

## Train

```bash
python classifier.py train --data demo_train.json --output artifacts
```

For real data, all input datasets use the same `intents` JSON structure:

```bash
python classifier.py train --data train.json --validation validation.json --output my_model
```

Without `--validation`, it makes a deterministic per-class 80/20 approximate split AFTER normalization and deduplication, requiring at least four unique normalized examples per class. Explicit validation is preferable for production; include every training class. Random splits still permit near-paraphrase leakage: separate whole wording/template families manually for a harder assessment.

The TF-IDF vocabulary and IDF weights are fitted on training data only. Checkpoint selection uses validation loss with patience-based early stopping. Training is seeded; results may still vary between software/hardware versions.

## Evaluate

```bash
python classifier.py evaluate --model artifacts --data demo_test.json --output test_report.json
```

Outputs accuracy, per-class precision/recall/F1/support, macro and weighted averages, a confusion matrix (rows = actual, columns = predicted), and individual errors. Test examples overlapping training OR validation after normalization are rejected. Labels absent from training are rejected. Do not tune hyperparameters or confidence thresholds using the final test set.

## Predict

```bash
python classifier.py predict --model artifacts --query "Show Overpayment details for the employee 998877"
```

Observed result from the supplied checkpoint:

```json
{
  "query": "Show Overpayment details for the employee 998877",
  "context": "case_details",
  "candidate_context": "case_details",
  "score": 0.998259,
  "needs_review": false,
  "reason": null
}
```

Optional rejection heuristic:

```bash
python classifier.py predict --model artifacts --threshold 0.8 --query "Show Overpayment details for the employee 998877"
```

0.8 is illustrative, not calibrated. Softmax scores are not guaranteed probabilities of correctness and can be high for out-of-domain inputs. Default threshold is 0 (ordinary closed-set classification); all-zero feature vectors are always rejected. A rejected prediction has `context: null`, `needs_review: true`, and a reason. Tune rejection with realistic validation and out-of-domain examples. An explicit `other` class is an additional option, not a universal unknown detector.

## Model and processing

`classifier.py` contains:

- `normalize`: lowercases, replaces all standalone digit sequences with `numbertoken`, expands `#`, and retains English alphabetic tokens. Intended for these English context labels; unsuitable without changes when numeric values affect the label or when multilingual/alphanumeric identifier semantics matter.
- `read_data`: validates the JSON, deduplicates normalized patterns, and rejects identical normalized inputs with different labels.
- `ContextClassifier`: TF-IDF input -> Linear(64) -> ReLU -> Dropout(0.2) -> Linear(number of contexts).
- `train`: learned word/unigram and bigram features, class-weighted cross-entropy, AdamW, and validation checkpoint selection.
- `evaluate`: held-out metrics and errors.
- `predict`: returns the label and model score.

scikit-learn performs TF-IDF encoding and metric calculations. PyTorch implements and trains the neural classifier from scratch. The encoder does not understand arbitrary unseen synonyms; broader real training examples are necessary. IDs are discarded for classification, not extracted. Preserve the original question separately if another component later extracts employee/case IDs.

## Saved files

- `artifacts/weights.pt`: model state dictionary, loaded using `weights_only=True`.
- `artifacts/metadata.json`: ordered context labels, TF-IDF vocabulary/IDF, model dimensions, seed, and hashes used for overlap checks.
- `artifacts/validation_report.json`: checkpoint validation results.
- `artifacts/test_report.json`: observed separate demo test results.

Keep model weights and metadata together. Retrain when adding labels or modifying normalization. Training to an existing output directory replaces that checkpoint. The demo uses dense vectors in memory and a vocabulary cap of 10,000 features; adapt batching for very large datasets.

## Actual verification

Tested on CPU with Python packages: torch 2.14.1+cu130, scikit-learn 1.8.0, numpy 2.3.5. These are the tested versions, while requirements.txt permits a wider compatible range.

- Training: 32 unique normalized training queries; 7 validation queries.
- Validation accuracy: 5/7 (71.43%).
- Separate demonstration test accuracy: 8/9 (88.89%); macro F1 about 0.8857.
- The test error was `Please fetch overpayment details for employee 998877`, predicted as `installment_details`. No changes were made to tune against this test result.
- Model reload and prediction succeeded with an unseen employee ID.
- Training the original single-context sample fails with a clear diagnostic.

These small, hand-written demonstration sets are not evidence of production accuracy. Collect representative real questions, negative examples, and confusing pairs. Keep number-only and capitalization-only variants together; they do not demonstrate language generalization.

## References

- PyTorch cross-entropy: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- PyTorch saving/loading: https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- scikit-learn text features: https://scikit-learn.org/stable/modules/feature_extraction.html
