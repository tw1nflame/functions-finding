"""Sphinx configuration for the library's HTML documentation."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

from parametric_regression import __version__  # noqa: E402

project = "Parametric Regression"
author = "Участники проекта Parametric Regression"
copyright = "2026, Участники проекта Parametric Regression"
release = __version__
version = release
language = "ru"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]
myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autoclass_content = "class"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
add_module_names = False
html_theme = "furo"
html_title = f"Parametric Regression {release}"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.svg"
html_show_sourcelink = False
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#176554",
        "color-brand-content": "#176554",
        "font-stack": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "font-stack--monospace": "'Cascadia Code', 'SFMono-Regular', Consolas, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#74d5ba",
        "color-brand-content": "#74d5ba",
    },
    "navigation_with_keys": True,
}
html_search_language = "ru"


def normalize_math_delimiters(app, docname, source):
    """Keep existing Markdown formulas readable in both repository and MyST."""
    if Path(app.env.doc2path(docname)).suffix == ".md":
        source[0] = source[0].replace("\\[\n", "$$\n").replace("\n\\]", "\n$$")


def setup(app):
    app.connect("source-read", normalize_math_delimiters)
    return {"parallel_read_safe": True}
