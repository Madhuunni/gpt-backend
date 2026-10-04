import json
from pathlib import Path

import torch


def load_metadata(directory, task):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("version") != 1 or metadata.get("task") != task:
        raise ValueError(f"Wrong artifact format/task in {directory}")
    vocab = metadata["vocab"]
    if vocab.get("<PAD>") != 0 or vocab.get("<UNK>") != 1 or sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Vocabulary indexes must be contiguous, with PAD=0 and UNK=1")
    if not 0 <= metadata["threshold"] <= 1 or metadata["max_tokens"] < 1:
        raise ValueError("Invalid inference settings in metadata")
    return metadata


def load_weights(model, directory):
    # Only trusted local artifacts; no remote model service or automatic download.
    # weights_only is supported by torch 2.2.1; no unsafe fallback is needed.
    state = torch.load(Path(directory) / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def save_weights(model, directory):
    """Portable tensor-only checkpoint; protocol 2 works with torch 2.2.1."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    torch.save(state, directory / "model.pt", pickle_protocol=2)
