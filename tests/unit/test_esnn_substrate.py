"""Unit tests for E-SNN thalamocortical substrate skeleton (C2.2)."""
from __future__ import annotations

import pytest

from kiki_oniric.substrates.esnn_thalamocortical import (
    ESNN_SUBSTRATE_NAME,
    ESNN_SUBSTRATE_VERSION,
    EsnnBackend,
    EsnnSubstrate,
    esnn_substrate_components,
)


def test_esnn_substrate_identity_constants() -> None:
    assert ESNN_SUBSTRATE_NAME == "esnn_thalamocortical"
    assert ESNN_SUBSTRATE_VERSION.startswith("C-v")


def test_esnn_backend_enum_has_two_options() -> None:
    backends = [EsnnBackend.NORSE, EsnnBackend.NXNET]
    values = [b.value for b in backends]
    assert values == ["norse", "nxnet"]


def test_esnn_substrate_instantiable_with_default_backend() -> None:
    substrate = EsnnSubstrate()
    assert substrate.backend == EsnnBackend.NORSE  # default


def test_esnn_substrate_backend_choice_persisted() -> None:
    substrate = EsnnSubstrate(backend=EsnnBackend.NXNET)
    assert substrate.backend == EsnnBackend.NXNET


def test_esnn_substrate_components_listed() -> None:
    components = esnn_substrate_components()
    expected_keys = {
        "primitives",
        "replay", "downscale", "restructure", "recombine",
        "finite", "topology",
        "runtime", "swap",
        "p_min", "p_equ", "p_max",
    }
    assert set(components.keys()) == expected_keys


def test_esnn_skeleton_ops_raise_not_implemented() -> None:
    substrate = EsnnSubstrate()
    with pytest.raises(NotImplementedError, match="C2.3"):
        substrate.replay_handler_factory()
    with pytest.raises(NotImplementedError, match="C2.3"):
        substrate.downscale_handler_factory()
