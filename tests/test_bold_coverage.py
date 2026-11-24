from energy.bold_calibration import (
    BOLDCalibrator,
    bold_signal_to_energy,
    calibrate_r_l_from_bold,
    compute_energy_with_ignition_spike,
    compute_landauer_energy_per_bit,
    estimate_bits_from_bold_energy,
    estimate_ignition_energy_spike,
    validate_energy_against_landauer,
)


def test_bold_basic_functions():
    # Landauer per bit
    e_bit = compute_landauer_energy_per_bit(310.0)
    assert e_bit > 0
    # BOLD to energy
    e = bold_signal_to_energy(bold_signal_change=2.0, baseline_energy=1.0)
    assert e > 0
    # Ignition spike
    spike = estimate_ignition_energy_spike(baseline_energy=1.0, spike_factor=1.1)
    assert spike == 1.1


def test_kappa_calibration():
    # Normal case
    kappa = calibrate_r_l_from_bold(bold_signal_change=2.0, bits_erased=10.0)
    assert kappa > 0
    # Zero bits
    kappa0 = calibrate_r_l_from_bold(bold_signal_change=2.0, bits_erased=0.0)
    assert kappa0 == 0.0
    # Zero temperature
    kappa_t0 = calibrate_r_l_from_bold(bold_signal_change=2.0, bits_erased=10.0, T=0.0)
    assert kappa_t0 == 0.0


def test_bits_from_bold():
    bits = estimate_bits_from_bold_energy(bold_signal_change=2.0, r_l=1000.0)
    assert bits > 0
    # Zero kappa
    bits0 = estimate_bits_from_bold_energy(bold_signal_change=2.0, r_l=0.0)
    assert bits0 == 0.0


def test_energy_with_spike():
    res = compute_energy_with_ignition_spike(baseline_bold=1.0, ignition_bold=2.0)
    assert res["total_energy_j"] > res["baseline_energy_j"]
    assert res["spike_factor"] == 2.0


def test_validate_landauer():
    # Satisfied
    res = validate_energy_against_landauer(measured_energy=1.0, bits_erased=1.0)
    assert res["satisfied"] is True
    # Violated
    res_bad = validate_energy_against_landauer(measured_energy=1e-30, bits_erased=1e6)
    assert res_bad["satisfied"] is False
    assert "violated" in res_bad["message"].lower()
    # Zero bits
    res0 = validate_energy_against_landauer(measured_energy=1.0, bits_erased=0.0)
    assert res0["satisfied"] is True


def test_bold_calibrator():
    cal = BOLDCalibrator()
    # Initial state
    assert cal.get_calibration_summary()["calibrated"] is False
    # Calibrate trial
    k1 = cal.calibrate_from_trial(baseline_bold=1.0, ignition_bold=2.0, estimated_bits=10.0)
    assert k1 > 0
    assert cal.calibrated_r_l == k1
    # Calibrate another
    k2 = cal.calibrate_from_trial(baseline_bold=1.0, ignition_bold=3.0, estimated_bits=5.0)
    assert k2 > 0
    assert cal.calibrated_r_l != k1  # Should be average
    # Summary
    summary = cal.get_calibration_summary()
    assert summary["calibrated"] is True
    assert summary["n_trials"] == 2
    # Validation
    v = cal.validate_against_landauer(1.0, 1.0)
    assert v["satisfied"] is True
    # Zero bits trial
    k0 = cal.calibrate_from_trial(1.0, 2.0, 0.0)
    assert k0 == 0.0
