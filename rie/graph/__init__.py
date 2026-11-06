from .workflow import ResearchWorkflow

__all__ = ["ResearchWorkflow", "build_langgraph", "run_langgraph"]


def __getattr__(name: str):
    if name in {"build_langgraph", "run_langgraph"}:
        from . import langgraph_workflow

        return getattr(langgraph_workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
