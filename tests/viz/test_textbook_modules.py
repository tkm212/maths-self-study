"""Parametrized tests that textbook viz modules ship non-empty content."""

from __future__ import annotations

from types import ModuleType

import pytest

from tests.viz.textbook_catalog import DEFINITION_IDS, DEFINITION_MODULES, THEOREM_IDS, THEOREM_MODULES


@pytest.mark.parametrize(("module", "name"), DEFINITION_MODULES, ids=DEFINITION_IDS)
def test_definition_module_nonempty(module: ModuleType, name: str):
    items = getattr(module, name)
    assert len(items) >= 1
    for term, definition in items:
        assert term.strip()
        assert len(definition.strip()) > 10


@pytest.mark.parametrize(("module", "name"), THEOREM_MODULES, ids=THEOREM_IDS)
def test_theorem_module_nonempty(module: ModuleType, name: str):
    items = getattr(module, name)
    assert len(items) >= 1
    for title, statement in items:
        assert title.strip()
        assert len(statement.strip()) > 10
