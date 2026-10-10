"""
Checks that the two descriptions of the package agree: setup.py (kept for compatibility with upstream)
and pyproject.toml (read by uv and the lockfile, and the one a build uses).

Key external dependencies: pytest; tomllib (Python >= 3.11, skipped otherwise).
"""

import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _setup_py_arguments() -> dict:
    """The keyword arguments of the setup() call in setup.py, read without running it."""
    with open(os.path.join(ROOT, "setup.py")) as file:
        tree = ast.parse(file.read())
    (call,) = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup"
    ]
    return {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in call.keywords
        if isinstance(keyword.value, (ast.Constant, ast.List, ast.Dict))
    }


def _normalised(requirements: list[str]) -> list[str]:
    return sorted(requirement.replace(" ", "") for requirement in requirements)


def test_setup_py_and_pyproject_agree():
    tomllib = pytest.importorskip("tomllib")
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as file:
        project = tomllib.load(file)["project"]
    setup = _setup_py_arguments()

    assert project["name"] == setup["name"]
    assert project["version"] == setup["version"]
    assert project["description"] == setup["description"]
    assert project["requires-python"] == setup["python_requires"]
    assert project["urls"]["Homepage"] == setup["url"]
    assert [(a["name"], a["email"]) for a in project["authors"]] == [(setup["author"], setup["author_email"])]
    assert _normalised(project["dependencies"]) == _normalised(setup["install_requires"])
    # pyproject.toml may define further extras (the experiment scripts); those of setup.py must match
    for extra, requirements in setup["extras_require"].items():
        assert _normalised(project["optional-dependencies"][extra]) == _normalised(requirements)
