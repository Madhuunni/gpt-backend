# PyTorch query ID extractor

Input: a query string. Output: a Python dictionary with exactly these string keys:

```json
{"case_id":"12345","employee_id":"","sr_no":""}
```

`json.dumps(result)` turns the dictionary into JSON text. Missing or rejected values
are empty strings. Training and inference run locally without LLM APIs or pretrained
model downloads. Installing PyTorch requires internet access unless installed offline.

## Quick start

Use Python 3.10+ with a compatible PyTorch build. In this project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python extractor.py --query "Show overpayment details for case ID 12345"
```

A trained CPU checkpoint is included in `artifacts/`. For a CPU-only PyTorch install,
you can instead use `python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu`.

```python
from extractor import IDExtractor
import json

extractor = IDExtractor("artifacts")  # Load once and reuse.
result = extractor.predict("Get me the recent letter generated for employee 001232")
print(json.dumps(result))
# {"case_id": "", "employee_id": "001232", "sr_no": ""}
```

## Retrain and evaluate

```bash
python prepare_data.py --input data/query_patterns.json
python train.py --epochs 12
python evaluate.py
python -m unittest test_extractor.py -v
```

For a CUDA-enabled installation, train with `python train.py --device cuda`.
Inference defaults to CPU; pass `device="cuda"` to IDExtractor if needed.

## Files

- `common.py`: offset-preserving tokenizer, labels, span validation, JSON decoder.
- `prepare_data.py`: reads the supplied intents JSON and generates span annotations,
  extra cue variants, randomized IDs, multi-ID examples and numeric negatives.
- `model.py`: embedding -> bidirectional LSTM -> four-class token classifier.
- `train.py`: training, best-checkpoint selection and validation-only threshold selection.
- `evaluate.py`: held-out exact JSON accuracy, per-field and micro entity metrics,
  and no-ID accuracy, including hard numeric negatives.
- `extractor.py`: reusable IDExtractor class and command-line inference.
- `test_extractor.py`: inference contract checks.
- `data/`: original supplied patterns, generated JSONL splits and data summary.
- `artifacts/`: checkpoint, metadata and actual training/evaluation reports.

## Why this is an extraction model

The earlier intent classifier answers which context a query belongs to. This model
answers which numeric tokens are case IDs, employee IDs or SR numbers. Each token
gets one of `O`, `case_id`, `employee_id`, `sr_no`. `O` means ordinary text.
All numeric tokens are represented as `<NUM>` to the model. It learns from the
surrounding words, not a fixed list of ID values. The decoder copies the original
number from the input, so new numbers and leading zeros are preserved.

Regex is used to bootstrap training annotations and to tokenize/validate numeric
candidates. The production predictor does **not** run the cue-labeling regex rules;
the neural model decides the numeric token's class. A no-numeric-token query returns
empty values immediately. This is a numeric-ID model, not a general named-entity model.

`metadata.json` stores the training vocabulary, label order, architecture settings,
and selected confidence threshold. It must stay with its matching `model.pt`.
The checkpoint contains a state dictionary and is loaded with `weights_only=True`.
Reference: https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

## Data and evaluation limitations

The attachment contains 3,200 patterns across `case_details` and `letter_generation`,
including 800 patterns without an ID. The intent `context` is not an entity label;
training labels are derived from explicit ID cues. The original attachment is copied
unchanged into `data/query_patterns.json`.

Rule-generated labels are weak supervision and should be reviewed. The preparation
script fails if any numeric source token is left unlabeled, rather than guessing its
meaning. Synthetic negatives are explicitly annotated as having no entities.

Splits group source wording after normalizing ID values and cue types. Generated
variants stay in their original group. This prevents identical template/value
variants crossing splits, but similar paraphrases can still occur in multiple splits.
Reported test scores measure this generated dataset, not arbitrary real-world language.
A manually labeled set of real queries is needed before deployment. Never use the test
set to choose epochs, thresholds or vocabulary. The current threshold is selected from
validation data only; softmax confidence is not a calibrated probability of correctness.

No-ID behavior is probabilistic when a query contains other numbers. Unknown phrasing,
negation, quoted IDs and ambiguous bare numbers can lead to errors even at high confidence.
Add representative reviewed examples and retrain as your queries evolve.

To label your own queries, use this JSONL schema (character offsets are end-exclusive):

```json
{"text":"case 00123","entities":[{"start":5,"end":10,"label":"case_id"}],"group":"my-template-family"}
{"text":"Show the latest letter","entities":[],"group":"my-no-id-family"}
```

Edit the generated splits directly or extend preparation. Running `prepare_data.py`
again overwrites the generated splits, so keep manual annotations separately. Assign
all related paraphrases/ID variants to one split. Training requires a `group` field;
evaluation accepts rows with just `text` and `entities`.

## Defined output behavior

- Numeric ASCII IDs only; no arbitrary maximum ID length. Alphanumeric, decimal and
  hyphenated tokens are unsupported and are not extracted as numeric substrings.
- If multiple ID types are found, all corresponding fields are filled.
- If multiple values of the same type are found, the first accepted mention wins.
  For workflows requiring every value, change the schema to arrays before deployment.
- If no numeric token is classified as an entity above the threshold, all fields are empty.
- Blank input returns all empty strings; non-string input raises TypeError.
- Queries over 512 tokens raise ValueError rather than silently truncating.
- This model cannot guarantee that every numeric non-ID is rejected. Choose a suitable
  threshold using real validation data; `IDExtractor(threshold=0.9)` overrides the saved one.

You can run this alongside your context classifier and combine its result with the
context label. The two models solve separate tasks.
