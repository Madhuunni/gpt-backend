"""Register additional tasks here, in execution order."""
from .graph import ExtraTask


def example_summary(state):
    return {"context": state["context"], "identifier_count": sum(bool(v) for v in state["identifiers"].values())}


# Uncomment to add a third graph task. It runs only for recognized contexts.
EXTRA_TASKS: list[ExtraTask] = [
    # ExtraTask("summarize_request", example_summary),
]
