from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


_template_env = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parent),
    autoescape=select_autoescape(enabled_extensions=("html", "xml")),
)
