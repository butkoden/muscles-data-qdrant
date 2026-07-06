# muscles-data-qdrant

Qdrant adapter package for `muscles-data`.

This package is intentionally separate from `muscles-data`: the core package
owns typed ports, resource runtime and diagnostics, while this package owns the
Qdrant-backed `VectorSearchPort` implementation.

## Usage

Register the factory in the project composition root:

```python
from muscles_data.catalog import DataAdapterCatalog
from muscles_data.ports import VectorSearchPort
from muscles_data.runtime import DataRuntime
from muscles_data_qdrant import QdrantVectorFactory

catalog = DataAdapterCatalog.with_defaults()
catalog.register(QdrantVectorFactory())

runtime = DataRuntime(config=config, catalog=catalog)
vector = runtime.require_port("vector.docs", VectorSearchPort)
```

Resource config stays in the project:

```yaml
data:
  resources:
    vector.docs:
      type: qdrant
      url: ${QDRANT_URL}
      api_key: ${QDRANT_API_KEY}
      collection: docs
      timeout: 3
      prefer_grpc: false
```

The adapter creates the Qdrant client lazily on vector search/write/delete,
explicit native access or `data.doctor`. Application code should use
`VectorSearchPort`; direct client access is only an advanced escape hatch with
`native_client: true`.

See `muscular-example/example_data_qdrant_1` for an executable example.
