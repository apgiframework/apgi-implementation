"""Model variants (S1-IMP-14.x) and the model registry (S1-IMP-12.x)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from model_registry import (
    MAPPINGS,
    REFERENCE_MODEL_ID,
    ModelVariant,
    bounds,
    canonical_name,
    check_against_registry,
    create_model,
    get_variant,
    list_variants,
    load_parameter_registry,
    pipeline_to_reduced,
    reduced_to_pipeline,
    register_variant,
    unregister_variant,
)
from models import (
    REDUCED_PARAMETER_NAMES,
    REDUCED_PARAMETERS,
    ApgiNull,
    ApgiReduced,
    ReducedParams,
    reduced_parameter_spec,
    simulate_reduced,
)
from models.reduced import LAM, ReducedState, learning_gain, reduced_step

ALL_IDS = [
    "apgi_agent",
    "apgi_extended",
    "apgi_hierarchical",
    "apgi_liquid",
    "apgi_null",
    "apgi_reduced",
    "apgi_reference",
]


def _stimulus(n: int = 80) -> np.ndarray:
    return np.sin(np.arange(n) * 0.15) + 0.3


class TestRegistry:
    def test_all_seven_variants_registered(self):
        assert [v.model_id for v in list_variants()] == ALL_IDS

    def test_single_authority_is_reduced(self):
        holders = [v.model_id for v in list_variants() if v.authority]
        assert holders == [REFERENCE_MODEL_ID] == ["apgi_reduced"]

    def test_second_authority_refused(self):
        class Rogue(ApgiNull):
            model_id = "apgi_rogue"
            authority = True

        with pytest.raises(ValueError, match="already holds reference authority"):
            register_variant(ModelVariant("apgi_rogue", Rogue, "x"))
        assert "apgi_rogue" not in [v.model_id for v in list_variants()]

    def test_duplicate_and_mismatched_ids_refused(self):
        with pytest.raises(ValueError, match="already registered"):
            register_variant(get_variant("apgi_null"))
        with pytest.raises(ValueError, match="class model_id"):
            register_variant(ModelVariant("wrong_id", ApgiNull, "x"))

    def test_register_and_unregister_non_authoritative(self):
        class Extra(ApgiNull):
            model_id = "apgi_extra"

        register_variant(ModelVariant("apgi_extra", Extra, "test"))
        try:
            assert isinstance(create_model("apgi_extra", seed=1), Extra)
        finally:
            unregister_variant("apgi_extra")
        with pytest.raises(KeyError):
            get_variant("apgi_extra")

    def test_unknown_model_lists_known(self):
        with pytest.raises(KeyError, match="apgi_reduced"):
            get_variant("nope")

    def test_describe_is_json_serialisable(self):
        import json

        for v in list_variants():
            json.dumps(v.describe())


@pytest.mark.parametrize("model_id", ALL_IDS)
class TestEveryVariant:
    def test_run_returns_required_keys_finite(self, model_id):
        out = create_model(model_id, seed=3).run(_stimulus(), 0.2 * _stimulus())
        assert set(out) >= {"S", "theta", "B", "p_ignite"}
        for v in out.values():
            assert v.shape == (80,) and np.all(np.isfinite(v))
        assert set(np.unique(out["B"])) <= {0.0, 1.0}
        assert np.all((out["p_ignite"] >= 0) & (out["p_ignite"] <= 1))

    def test_same_seed_same_trajectory(self, model_id):
        a = create_model(model_id, seed=11).run(_stimulus())
        b = create_model(model_id, seed=11).run(_stimulus())
        for k in a:
            assert np.array_equal(a[k], b[k])

    def test_reset_restores_initial_behaviour(self, model_id):
        m = create_model(model_id, seed=5)
        first = m.run(_stimulus(40))
        m.reset(5)
        second = m.run(_stimulus(40))
        assert np.array_equal(first["S"], second["S"])
        assert np.array_equal(first["B"], second["B"])

    def test_x_i_shape_mismatch_rejected(self, model_id):
        with pytest.raises(ValueError, match="shape"):
            create_model(model_id, seed=1).run(np.zeros(5), np.zeros(4))


class TestReducedParams:
    def test_defaults_inside_support(self):
        ReducedParams()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"alpha": 0.0},
            {"alpha": 1.5},
            {"pi": 0.05},
            {"pi": 11},
            {"theta_base": 0.1},
            {"beta": 2.0},
            {"beta": float("nan")},
        ],
    )
    def test_out_of_support_rejected(self, kwargs):
        with pytest.raises(ValueError, match="outside support"):
            ReducedParams(**kwargs)

    def test_noise_must_be_positive(self):
        with pytest.raises(ValueError, match="sigma"):
            ReducedParams(sigma_r=0.0)

    def test_from_mapping_unknown_key(self):
        with pytest.raises(KeyError, match="unknown apgi_reduced parameters"):
            ReducedParams.from_mapping({"nonsense": 1})

    def test_estimable_roundtrip(self):
        p = ReducedParams(alpha=0.2, pi=2.0, theta_base=0.4, beta=-0.1)
        q = p.with_estimable(p.estimable())
        assert q == p
        with pytest.raises(ValueError, match="expected 4 values"):
            p.with_estimable([1.0])

    def test_parameter_specs_consistent(self):
        assert [p.name for p in REDUCED_PARAMETERS] == list(REDUCED_PARAMETER_NAMES)
        for spec in REDUCED_PARAMETERS:
            assert spec.lower <= spec.plausible_lower < spec.plausible_upper <= spec.upper
            assert spec.contains(spec.nominal)
        with pytest.raises(KeyError):
            reduced_parameter_spec("zeta")


class TestReducedModel:
    def test_learning_gain_depends_only_on_product(self):
        assert learning_gain(0.2, 2.0) == pytest.approx(learning_gain(0.4, 1.0))
        assert learning_gain(0.5, 1.0) == pytest.approx(1 / 3)

    def test_first_trial_hand_derived(self):
        # x=1, x̂=0 ⇒ ε=1; drive = tanh(2·1.5·1)+0.05; S = λ·drive.
        state, rec = reduced_step(ReducedParams(), ReducedState(theta=0.55), 1.0, hard=True)
        drive = math.tanh(3.0) + 0.05
        assert rec["S"] == pytest.approx(LAM * drive, rel=1e-12)
        assert rec["eps"] == 1.0 and rec["B"] == 0.0
        k = 0.3 * 1.5 / (1 + 0.3 * 1.5)
        assert state.xhat == pytest.approx(k)

    def test_stochastic_step_requires_draw(self):
        with pytest.raises(ValueError, match="uniform draw"):
            reduced_step(ReducedParams(), ReducedState(), 1.0)

    def test_forced_ignition_changes_future(self):
        p = ReducedParams()
        x = _stimulus(30)
        a = simulate_reduced(p, x, B_forced=np.zeros(30, dtype=int))
        b = simulate_reduced(p, x, B_forced=np.r_[1, np.zeros(29, dtype=int)])
        assert b.theta[1] > a.theta[1]  # refractory boost
        assert b.S[1] < a.S[1]  # reset

    def test_forced_length_checked(self):
        with pytest.raises(ValueError, match="same length"):
            simulate_reduced(ReducedParams(), np.zeros(5), B_forced=np.zeros(4, dtype=int))

    def test_hard_mode_is_deterministic_and_rng_free(self):
        p = ReducedParams(theta_base=0.3)
        x = np.tile([0.8, -0.8], 40)
        a = simulate_reduced(p, x, hard=True)
        b = simulate_reduced(p, x, hard=True)
        assert np.array_equal(a.B, b.B) and a.B.sum() > 0

    def test_step_wrapper_matches_simulate(self):
        x = _stimulus(60)
        tr = simulate_reduced(ReducedParams(), x, rng=9)
        run = ApgiReduced(seed=9).run(x)
        assert np.array_equal(run["S"], tr.S) and np.array_equal(run["B"], tr.B)

    def test_state_snapshot_roundtrip_continues_identically(self):
        x = _stimulus(60)
        m = ApgiReduced(seed=2)
        for v in x[:25]:
            m.step(float(v))
        snap = m.get_state()
        tail_a = [m.step(float(v))["S"] for v in x[25:]]
        m2 = ApgiReduced(seed=999)
        m2.set_state(snap)
        tail_b = [m2.step(float(v))["S"] for v in x[25:]]
        assert tail_a == tail_b

    def test_hard_flag_and_unknown_param(self):
        assert ApgiReduced({"hard": True}).hard is True
        with pytest.raises(KeyError):
            ApgiReduced({"bogus": 1})


class TestNullAndPipelineModels:
    def test_null_rate_validated_and_respected(self):
        with pytest.raises(ValueError, match="rate"):
            ApgiNull({"rate": 1.5})
        out = ApgiNull({"rate": 0.3}, seed=1).run(np.zeros(4000))
        assert out["B"].mean() == pytest.approx(0.3, abs=0.03)
        assert np.all(out["S"] == 0.0)

    def test_null_state_roundtrip(self):
        a = ApgiNull(seed=4)
        for _ in range(10):
            a.step(0.0)
        snap = a.get_state()
        nxt = [a.step(0.0)["B"] for _ in range(20)]
        b = ApgiNull(seed=0)
        b.set_state(snap)
        assert [b.step(0.0)["B"] for _ in range(20)] == nxt

    def test_pipeline_models_report_no_snapshot(self):
        for mid in (
            "apgi_reference",
            "apgi_extended",
            "apgi_hierarchical",
            "apgi_liquid",
            "apgi_agent",
        ):
            m = create_model(mid, seed=1)
            assert m.get_state() is None
            with pytest.raises(NotImplementedError):
                m.set_state({})

    def test_pipeline_config_override(self):
        base = create_model("apgi_reference", seed=1).run(_stimulus())
        other = create_model(
            "apgi_reference", {"config": {"theta_0": 0.8, "theta_base": 0.8}}, seed=1
        ).run(_stimulus())
        assert not np.array_equal(base["theta"], other["theta"])


class TestParameterMapping:
    def test_alpha_is_not_alpha_e(self):
        by = {m.symbol: m for m in MAPPINGS}
        assert by["α"].pipeline == "kappa_e" and "alpha_e" not in {m.pipeline for m in MAPPINGS}
        assert by["Π"].pipeline is None

    def test_translation_roundtrip(self):
        pipe = reduced_to_pipeline({"alpha": 0.3, "pi": 1.0, "theta_base": 0.5, "beta": 0.1})
        assert pipe == {"kappa_e": 0.3, "theta_base": 0.5, "beta": 0.1}  # pi has no counterpart
        assert pipeline_to_reduced(pipe) == {"alpha": 0.3, "theta_base": 0.5, "beta": 0.1}

    def test_aliases_resolve(self):
        assert canonical_name("beta_da") == "beta"
        assert canonical_name("theta_0") == "theta_base"
        assert canonical_name("unmapped") == "unmapped"

    def test_bounds(self):
        assert bounds("pi") == (0.1, 10.0)
        assert bounds("beta_da") == (-1.0, 1.0)
        with pytest.raises(KeyError):
            bounds("kappa")

    def test_registry_drift_detection(self, tmp_path):
        good = {
            p.name: {"lower": p.lower, "upper": p.upper, "symbol": p.symbol}
            for p in REDUCED_PARAMETERS
        }
        assert check_against_registry(good) == []
        bad = {**good, "pi": {"lower": 0.2, "upper": 10.0, "symbol": "Π"}}
        del bad["beta"]
        bad["extra"] = {"lower": 0, "upper": 1}
        problems = check_against_registry(bad)
        assert any("pi.lower" in p for p in problems)
        assert any("beta: missing" in p for p in problems)
        assert any("extra: in registry" in p for p in problems)

    def test_load_registry(self, tmp_path):
        import json

        f = tmp_path / "r.json"
        f.write_text(json.dumps({"parameters": {"alpha": {"lower": 0.001, "upper": 1.0}}}))
        assert load_parameter_registry(f)["alpha"]["upper"] == 1.0
        f.write_text("[]")
        with pytest.raises(ValueError, match="parameters"):
            load_parameter_registry(f)
