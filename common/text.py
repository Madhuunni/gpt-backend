import re


def tokenize(text):
    """Return text spans so extracted IDs preserve case and leading zeros."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]


def normalize(token):
    if token.isdecimal():
        return "<NUM>"
    if any(c.isdigit() for c in token) and token.isalnum():
        return "<ID>"
    return token.lower()


def encode(tokens, vocab):
    return [vocab.get(normalize(t), 1) for t, _, _ in tokens]
