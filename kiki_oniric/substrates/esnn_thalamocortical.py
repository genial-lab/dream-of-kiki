"""E-SNN thalamocortical substrate skeleton (cycle-2 C2.2).

Second substrate for dreamOfkiki, validating DR-3 substrate-
agnosticism empirically alongside MLX kiki-oniric.

Backend choice :
- `EsnnBackend.NORSE` (default) : PyTorch-based spiking neural
  network simulator. Install via `pip install norse`. Simulation
  on any GPU/CPU — no special hardware needed. Recommended for
  cycle-2 validation unless Loihi-2 access is granted.
- `EsnnBackend.NXNET` : Intel Loihi-2 native NxSDK/NxNet runtime.
  Requires hardware access via Intel NRC partnership. Not
  installed by default.

The skeleton defines the substrate identity, backend enum, and
stub factory methods. Real operations wiring (replay/downscale/
restructure/recombine as spike-rate dynamics) lands in C2.3.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md
§6.2 (DR-3 Conformance Criterion) ; cycle-2 plan
docs/superpowers/plans/2026-04-18-dreamofkiki-cycle2-atomic.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


ESNN_SUBSTRATE_NAME = "esnn_thalamocortical"
ESNN_SUBSTRATE_VERSION = "C-v0.5.0+STABLE"


class EsnnBackend(str, Enum):
    """Backend choice for the E-SNN substrate."""

    NORSE = "norse"
    NXNET = "nxnet"


@dataclass
class EsnnSubstrate:
    """E-SNN thalamocortical substrate — skeleton (C2.2).

    Operations wiring deferred to C2.3. Instantiation + backend
    choice persisted here ; factory methods raise
    NotImplementedError pointing to C2.3 for the concrete
    handlers.
    """

    backend: EsnnBackend = EsnnBackend.NORSE

    def replay_handler_factory(self):
        """Stub for C2.3 — E-SNN replay op as spike-rate replay."""
        raise NotImplementedError(
            f"E-SNN replay handler not yet implemented "
            f"(backend={self.backend.value!r}, see cycle-2 task C2.3)"
        )

    def downscale_handler_factory(self):
        """Stub for C2.3 — E-SNN downscale via synaptic scaling."""
        raise NotImplementedError(
            f"E-SNN downscale handler not yet implemented "
            f"(backend={self.backend.value!r}, see cycle-2 task C2.3)"
        )

    def restructure_handler_factory(self):
        """Stub for C2.3 — E-SNN restructure via topology edits."""
        raise NotImplementedError(
            f"E-SNN restructure handler not yet implemented "
            f"(backend={self.backend.value!r}, see cycle-2 task C2.3)"
        )

    def recombine_handler_factory(self):
        """Stub for C2.3 — E-SNN recombine via spike patterns."""
        raise NotImplementedError(
            f"E-SNN recombine handler not yet implemented "
            f"(backend={self.backend.value!r}, see cycle-2 task C2.3)"
        )


def esnn_substrate_components() -> dict[str, str]:
    """Return the canonical map of E-SNN substrate components.

    Mirrors `mlx_kiki_oniric.mlx_substrate_components()` keys so
    the DR-3 Conformance Criterion test suite can parametrize
    over both substrates. Cycle-2 C2.3 replaces skeleton stubs
    with actual paths.
    """
    return {
        # Substrate-agnostic primitives (shared with MLX)
        "primitives": "kiki_oniric.core.primitives",
        # 4 operations (skeleton now, real wiring C2.3)
        "replay": "kiki_oniric.substrates.esnn_thalamocortical",
        "downscale": "kiki_oniric.substrates.esnn_thalamocortical",
        "restructure": "kiki_oniric.substrates.esnn_thalamocortical",
        "recombine": "kiki_oniric.substrates.esnn_thalamocortical",
        # 2 invariant guards (substrate-agnostic, shared)
        "finite": "kiki_oniric.dream.guards.finite",
        "topology": "kiki_oniric.dream.guards.topology",
        # Runtime + swap (substrate-agnostic, shared)
        "runtime": "kiki_oniric.dream.runtime",
        "swap": "kiki_oniric.dream.swap",
        # 3 profiles (substrate-agnostic wrappers, shared)
        "p_min": "kiki_oniric.profiles.p_min",
        "p_equ": "kiki_oniric.profiles.p_equ",
        "p_max": "kiki_oniric.profiles.p_max",
    }
