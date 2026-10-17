"""Coverage tests for main.py entry point and CLI commands."""

import json
import os
import subprocess
import sys
from unittest.mock import patch

from main import analyze_signal_statistics, main, show_info


def test_main_info_command():
    """Test 'python main.py info' or 'python main.py' with info arg."""
    # We can test the function directly and also via CLI
    with patch("main.show_info") as mock_show:
        with patch("sys.argv", ["main.py", "info"]):
            exit_code = main()
            assert exit_code == 0
            mock_show.assert_called_once()


def test_show_info_execution(capsys):
    """Test show_info output."""
    show_info()
    captured = capsys.readouterr()
    assert "APGI: Allostatic Precision-Gated Ignition" in captured.out
    assert "Version: 1.1.0" in captured.out


def test_analyze_signal_statistics_exception():
    """Test analyze_signal_statistics exception handling (lines 334-335)."""
    # Create a signal that will cause estimate_hurst_robust to fail
    # We can mock estimate_hurst_robust to raise an exception
    with patch("main.estimate_hurst_robust", side_effect=Exception("Simulated Hurst failure")):
        signal = [1.0] * 300  # Long enough to trigger Hurst estimation
        stats = analyze_signal_statistics(signal, label="TestFailure")
    # Check if warning was logged (it goes to logger, but we can check if it returns stats without hurst_exponent if that's the logic)
    assert "hurst_exponent" not in stats
    # The logger warning is: logger.warning("hurst_estimation_failed", label=label, error=str(e))


def test_main_full_execution_demo():
    """Test main with --demo to cover more lines."""
    with patch("sys.argv", ["main.py", "--demo", "--steps", "10"]):
        exit_code = main()
        assert exit_code == 0


def test_main_full_execution_multiscale():
    """Test main with --multiscale to cover more lines."""
    with patch("sys.argv", ["main.py", "--multiscale", "--steps", "10", "--levels", "3"]):
        exit_code = main()
        assert exit_code == 0


def test_main_output_save(tmp_path):
    """Test saving results to a file."""
    output_file = tmp_path / "results.json"
    with patch("sys.argv", ["main.py", "--steps", "10", "--output", str(output_file)]):
        exit_code = main()
        assert exit_code == 0
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
            assert "history" in data


def test_main_module_execution():
    """Test the if __name__ == '__main__': block using subprocess."""
    # This covers line 562
    result = subprocess.run(
        [sys.executable, "main.py", "--steps", "2"], capture_output=True, text=True, cwd=os.getcwd()
    )
    assert result.returncode == 0


def test_main_ne_on_threshold_auto_adjust():
    """Test auto-adjustment of gamma_ne in ne-on-threshold mode."""
    with patch("sys.argv", ["main.py", "--ne-on-threshold", "--steps", "2"]):
        # We need to capture the logger or just check the exit code
        exit_code = main()
        assert exit_code == 0


def test_main_keyboard_interrupt():
    """Test KeyboardInterrupt handling."""
    with patch("main.run_standard_pipeline", side_effect=KeyboardInterrupt):
        with patch("sys.argv", ["main.py", "--steps", "10"]):
            exit_code = main()
            assert exit_code == 130


def test_main_generic_exception():
    """Test generic exception handling."""
    with patch("main.run_standard_pipeline", side_effect=ValueError("Test error")):
        with patch("sys.argv", ["main.py", "--steps", "10"]):
            exit_code = main()
            assert exit_code == 1
