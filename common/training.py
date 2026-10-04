"""Shared training mechanics; the two task packages expose separate commands."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from common.text import encode, normalize, tokenize
from common.artifacts import save_weights

TAGS = ["O", "employee_id", "case_id", "sr_no"]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def train_task(task):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--output", default="artifacts/" + task)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    torch.manual_seed(17)
    torch.set_num_threads(2)
    train_rows, valid_rows = read_rows(args.data), read_rows(args.validation)
    if not train_rows or not valid_rows or args.epochs < 1:
        raise ValueError("Nonempty training/validation sets and positive epochs are required")
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token in sorted({normalize(t) for r in train_rows for t, _, _ in tokenize(r["query"])}):
        if token not in vocab:
            vocab[token] = len(vocab)
    labels = sorted({r.get("intent", r.get("context", "unsupported")) or "unsupported" for r in train_rows})
    if task == "intent_classification":
        from intent_classification.model import IntentModel
        if "unsupported" not in labels:
            raise ValueError("Include unsupported/unknown examples with intent='unsupported'")
        model = IntentModel(len(vocab), len(labels))
    else:
        from identifier_extraction.model import IdentifierModel
        model = IdentifierModel(len(vocab), len(TAGS))

    def example(row):
        tokens = tokenize(row["query"])
        if not tokens or len(tokens) > 100:
            raise ValueError("Training queries must contain 1–100 tokens")
        if task == "intent_classification":
            target = labels.index(row.get("intent", row.get("context", "unsupported")) or "unsupported")
        else:
            target = [0] * len(tokens)
            for entity in row["entities"]:
                matches = [i for i, (_, a, b) in enumerate(tokens) if a == entity["start"] and b == entity["end"]]
                if len(matches) != 1 or entity["field"] not in TAGS[1:]:
                    raise ValueError("Entity must cover exactly one complete token with a supported field")
                target[matches[0]] = TAGS.index(entity["field"])
        return torch.tensor(encode(tokens, vocab)), torch.tensor(target)

    def collate(batch):
        xs, ys = zip(*batch)
        targets = torch.stack(ys) if task == "intent_classification" else pad_sequence(ys, batch_first=True, padding_value=-100)
        return pad_sequence(xs, batch_first=True), torch.tensor([len(x) for x in xs]), targets

    train_loader = DataLoader([example(r) for r in train_rows], batch_size=64, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader([example(r) for r in valid_rows], batch_size=64, collate_fn=collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for epoch in range(args.epochs):
        model.train()
        total = 0
        for x, lengths, targets in train_loader:
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = torch.nn.functional.cross_entropy(logits if task == "intent_classification" else logits.reshape(-1,len(TAGS)), targets if task == "intent_classification" else targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            total += loss.item()
        print(f"{task} epoch {epoch+1}: loss={total/len(train_loader):.4f}", flush=True)
    model.eval()
    correct = 0
    with torch.inference_mode():
        for x, lengths, targets in valid_loader:
            predictions = model(x, lengths).argmax(-1)
            correct += ((predictions == targets).sum().item() if task == "intent_classification"
                        else ((predictions == targets) | (targets == -100)).all(dim=1).sum().item())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_weights(model, output)
    metadata = {"version":1,"task":task,"vocab":vocab,"max_tokens":100,"max_unknown_fraction":.45,
                "threshold":.85 if task == "intent_classification" else .8}
    metadata.update({"labels":labels,"reject_labels":["unsupported"]} if task == "intent_classification" else {"tags":TAGS})
    (output / "metadata.json").write_text(json.dumps(metadata,indent=2))
    report = {"examples":len(valid_rows),"metric":"intent_accuracy" if task == "intent_classification" else "exact_token_tag_sequence_accuracy",
              "value":correct/len(valid_rows),"note":"Raw model predictions on supplied development validation; not calibrated or a production accuracy guarantee."}
    (output / "validation_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
