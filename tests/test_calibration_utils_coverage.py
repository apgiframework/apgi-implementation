import pytest  # noqa: F401

from energy.calibration_utils import (
    calibrate_for_realistic_r_l,
    create_realistic_calibrator,
    demonstrate_calibration_range,
)


def test_calibrate_for_realistic_kappa():
    res = calibrate_for_realistic_r_l(target_r_l=1000.0)
    assert res["target_r_l"] == 1000.0
    assert res["calibrated_conversion_factor"] > 0
    assert res["landauer_minimum_j_per_bit"] > 0


def test_create_realistic_calibrator():
    cal, info = create_realistic_calibrator(target_efficiency=500.0)
    assert info["target_r_l"] == 500.0
    assert cal.conversion_factor == info["calibrated_conversion_factor"]


def test_demonstrate_calibration_range(capsys):
    demonstrate_calibration_range()
    captured = capsys.readouterr()
    assert "Theoretical minimum" in captured.out
    assert "Pathological" in captured.out
