"""Pipeline-backed APGI variants.

These variants wrap :class:`pipeline.APGIPipeline` with fixed configuration
presets. They carry no reference authority (decision D14): ``apgi_reference`` is
a fixture-conformant reading of the specification, and the others are
extension-specific or exploratory. Only ``apgi_reduced`` is qualified.

Checkpointing is by deterministic replay (the pipeline holds live generators,
reservoirs and oscillators that have no stable serialised form), so
:meth:`get_state` returns ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from config import CONFIG
from core.rng import resolve_rng
from pipeline import APGIPipeline

from .base import APGIModel

__all__ = [
    "PipelineModel",
    "ApgiReference",
    "ApgiExtended",
    "ApgiHierarchical",
    "ApgiLiquid",
    "ApgiAgent",
    "ApgiNull",
]


class PipelineModel(APGIModel):
    """Base class: an :class:`APGIPipeline` plus a preset of config overrides."""

    #: Config overrides applied on top of :data:`config.CONFIG`.
    preset: ClassVar[dict[str, Any]] = {}
    parameter_names: ClassVar[tuple[str, ...]] = ("config",)

    def __init__(self, params: Mapping[str, Any] | None = None, seed: int | None = None):
        super().__init__(params, seed)

    def _build_config(self, seed: int | None) -> dict[str, Any]:
        cfg = {**CONFIG, **self.preset}
        cfg.update(self._params.get("config", {}))
        if seed is not None:
            cfg["seed"] = seed
        return cfg

    def reset(self, seed: int | None = None) -> None:
        self.seed = seed
        self.pipeline = APGIPipeline(self._build_config(seed))

    def step(self, x_e: float, x_i: float | None = None) -> dict[str, float]:
        rec = self.pipeline.step(x_e=x_e, x_i=0.0 if x_i is None else x_i)
        out = {k: float(rec[k]) for k in ("S", "theta", "B", "p_ignite")}
        out["pi_e"] = float(rec["pi_e"])
        return out


class ApgiReference(PipelineModel):
    """Fixture-conformant reading of the spec: the pipeline at canonical defaults."""

    model_id: ClassVar[str] = "apgi_reference"
    status: ClassVar[str] = "fixture-conformant"
    preset: ClassVar[dict[str, Any]] = {}


class ApgiExtended(PipelineModel):
    """Exploratory extension: somatic precision, NE threshold pathway, circadian drift."""

    model_id: ClassVar[str] = "apgi_extended"
    status: ClassVar[str] = "exploratory"
    preset: ClassVar[dict[str, Any]] = {
        "use_somatic_precision": True,
        "ne_on_precision": False,
        "ne_on_threshold": True,
        "gamma_ne": 0.01,
        "use_circadian_modulation": True,
    }


class ApgiHierarchical(PipelineModel):
    """Extension-specific: multi-timescale hierarchy with precision ODE."""

    model_id: ClassVar[str] = "apgi_hierarchical"
    status: ClassVar[str] = "extension-specific"
    preset: ClassVar[dict[str, Any]] = {"hierarchical_mode": "advanced"}


class ApgiLiquid(PipelineModel):
    """Extension-specific: liquid-state-machine reservoir layer."""

    model_id: ClassVar[str] = "apgi_liquid"
    status: ClassVar[str] = "extension-specific"
    preset: ClassVar[dict[str, Any]] = {"use_reservoir": True, "reservoir_size": 50}


class ApgiAgent(PipelineModel):
    """Computational comparison: perception-action loop (active inference)."""

    model_id: ClassVar[str] = "apgi_agent"
    status: ClassVar[str] = "computational-comparison"
    preset: ClassVar[dict[str, Any]] = {"use_active_inference": True}


class ApgiNull(APGIModel):
    """Benchmark null: ignition with no signal accumulation or threshold dynamics.

    Ignites with a constant probability ``rate`` independent of the input, so
    any structure an APGI variant explains beyond it is attributable to the
    mechanism rather than to base rates. ``S`` and ``theta`` are fixed.
    """

    model_id: ClassVar[str] = "apgi_null"
    status: ClassVar[str] = "benchmark"
    parameter_names: ClassVar[tuple[str, ...]] = ("rate",)

    def __init__(self, params: Mapping[str, Any] | None = None, seed: int | None = None):
        rate = float((params or {}).get("rate", 0.1))
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate
        super().__init__({"rate": rate}, seed)

    def reset(self, seed: int | None = None) -> None:
        self.seed = seed
        self._rng = resolve_rng(seed)

    def step(self, x_e: float, x_i: float | None = None) -> dict[str, float]:
        B = float(self._rng.random() < self.rate)
        return {"S": 0.0, "theta": 1.0, "B": B, "p_ignite": self.rate}

    def get_state(self) -> dict[str, Any]:
        st = self._rng.bit_generator.state
        return {
            "rng": {
                "bit_generator": st["bit_generator"],
                "state": {"state": int(st["state"]["state"]), "inc": int(st["state"]["inc"])},
                "has_uint32": int(st["has_uint32"]),
                "uinteger": int(st["uinteger"]),
            }
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        self._rng.bit_generator.state = state["rng"]
