# Supplementary material — manifest

Built by `python3 -m anonymize --build` (or `--build-dir`, which writes the
same file set unpacked). Everything below is stated so that the omissions are
visible rather than silent.

* **108 files** included.
* **Excluded**: caches (`__pycache__`, `.pytest_cache`), datasets
  (`data/`, `data_federated/`, FineWeb shards), the raw per-run trees
  (`results/federated/`, `results/synthetic/`, `results_old/`) and the
  unpacked export bundles, model checkpoints (`*.pt`), and stray `*.log`
  and `*.stackdump` files.
* **Included from `results/`**: the three export archives
  (`synthetic_results.zip`, `article_export.tar.gz`,
  `federated_export_results.zip`) and `nanogpt/`. The archives are what
  `REPRODUCE.md` §§3-5 plot from, so every CIFAR, federated and synthetic
  table and figure redraws from this bundle without a GPU. Each is the run
  set its exporter chose: the federated one excludes runs made under a
  superseded sign convention and lists each exclusion in its
  `MANIFEST.json`. The nanoGPT logs are included because they cost an
  8xH100 node and are the one artefact nothing here can regenerate.
* **Notebook outputs stripped**: 0 notebook(s).
  Source cells are untouched. Outputs are removed because they carry
  absolute paths and hostnames from the machine that ran them.

## Where to start

`README.md` is the map; `REPRODUCE.md` has the exact command for every table
and figure in the paper. `python3 -m tests.test_code` runs the CPU test suite
in about a minute and needs no GPU and no downloads.

## Anonymity

This tree was scanned before it was packaged, for author names, emails,
absolute paths containing usernames, the project's own repository URL, and
ORCIDs — inside the export archives as well as in the source. Upstream
citations (e.g. the modded-nanogpt repository this port builds on) are
deliberately kept: removing them would misattribute the work.

The scanner, its tests and its write-up are **not** in this bundle. Between
them they hold the list of names to look for and a record of what once
leaked, so shipping them would undo the scan. Nothing else references them,
and the suite above has no import that fails to resolve as a result.

**Git revisions are redacted.** The exporters stamp the commit each run was
made at into `metrics.json`, `configs.json`, `environment.json` and each
`MANIFEST.json`; in the archives above every such field now reads
`<redacted for review>`. A revision resolves to nothing without the
repository, which is not part of this bundle, so it bought reproducibility
nothing here and identified the authors to anyone who searched for it.
Everything else in the archives is exactly what the exporter wrote.
