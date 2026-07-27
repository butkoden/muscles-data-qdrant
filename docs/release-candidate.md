# `muscles-data-qdrant` RC checklist

The package ships the Qdrant implementation of `VectorSearchPort`.
The dependency on `muscles-data` is versioned as `>=0.1.0,<1.0.0`.

Before publishing a GitHub Release, run:

```bash
PYTHONPATH=../muscles-data/src:src python -m pytest -q
python -m build --wheel --sdist
```

The integration scenario is enabled with `MUSCLES_DATA_INTEGRATION=1` and a
running Qdrant service configured through `QDRANT_URL`. The adapter validates
vector dimensions and never generates embeddings itself.

The PyPI workflow publishes only after a GitHub Release is published. It uses
the versioned `muscles-data` dependency and trusted publishing.
