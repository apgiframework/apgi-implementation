# Changelog

All notable changes to `apgi` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-10-17

Release aligned with `apgi-framework` 1.1.0.

- `pyproject.toml`, `CITATION.cff`, `README.md` (BibTeX) and `main.py`: version 1.1.0. The CLI reads
  its version from a single `APGI_VERSION` constant instead of two hard-coded strings.
- Project name unified as Allostatic Precision-Gated Ignition. `main.py` previously said
  "Adaptive Precision Gated Ignition" in its docstring and `--help` text.
- `pyproject.toml`: removed the `orcid` key from `project.authors`. PEP 621 allows only `name` and
  `email`, so setuptools rejected the file and `pip install -e .` failed. The ORCID stays in
  `CITATION.cff`.
- Authorship: `pyproject.toml` and `CITATION.cff` name Dimitry Pesochin (with ORCID in
  `CITATION.cff`) in place of the generic "APGI Team" / "APGI Framework contributors".
- `requirements.txt` added with the same bounds as `pyproject.toml`; `numpy` capped at `<2.0`
  for reproducibility of the published figures.
- README: citation and archival sections revised.
- `tests/test_release_metadata.py` (new): fails if the version in `pyproject.toml`,
  `CITATION.cff`, the README BibTeX, this changelog and `main.py` disagree.

## [1.0.0] - 2026-07-29

First DOI-archived release ([10.5281/zenodo.21632264](https://doi.org/10.5281/zenodo.21632264)).
