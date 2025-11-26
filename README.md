# RuVector

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.77%2B-orange.svg)](https://www.rust-lang.org)
[![npm](https://img.shields.io/npm/v/ruvector-gnn.svg)](https://www.npmjs.com/package/ruvector-gnn)

**A vector database that learns.** Store embeddings, query with Cypher, and let the index improve itself through Graph Neural Networks.

## What Problem Does RuVector Solve?

Traditional vector databases just store and search. When you ask "find similar items," they return results but never get smarter.

**RuVector is different:**

1. **Store vectors** like any vector DB (embeddings from OpenAI, Cohere, etc.)
2. **Query with Cypher** like Neo4j (`MATCH (a)-[:SIMILAR]->(b) RETURN b`)
3. **The index learns** — GNN layers make search results improve over time

Think of it as: **Pinecone + Neo4j + PyTorch** in one Rust package.

## Quick Start

### Node.js / Browser

```bash
# Install
npm install ruvector-gnn

# Or try instantly with npx
npx ruvector-gnn --demo
```

```javascript
const { RuvectorLayer, TensorCompress, differentiableSearch } = require('ruvector-gnn');

// Create a GNN layer (128-dim input, 256 hidden, 4 attention heads)
const layer = new RuvectorLayer(128, 256, 4, 0.1);

// Your embeddings (from OpenAI, Cohere, or any model)
const queryEmbedding = new Float32Array([0.1, 0.2, ...]);
const documentEmbeddings = [
  new Float32Array([0.15, 0.22, ...]),
  new Float32Array([0.8, 0.1, ...]),
  // ... more documents
];

// Find top 10 similar with soft attention (differentiable!)
const { indices, weights } = differentiableSearch(queryEmbedding, documentEmbeddings, 10, 0.07);

console.log('Most similar:', indices);
console.log('Confidence:', weights);
```

### Rust

```bash
cargo add ruvector-graph ruvector-gnn
```

```rust
use ruvector_graph::{GraphDB, NodeBuilder, EdgeBuilder};
use ruvector_gnn::{RuvectorLayer, differentiable_search};

// Create graph database
let db = GraphDB::new();

// Add nodes with embeddings
let doc1 = NodeBuilder::new("doc1")
    .label("Document")
    .property("text", "Machine learning basics")
    .property("embedding", vec![0.1, 0.2, 0.3])
    .build();
db.create_node(doc1)?;

// Query with Cypher
let results = db.execute("MATCH (d:Document) WHERE d.text CONTAINS 'learning' RETURN d")?;

// Use GNN for smarter search
let layer = RuvectorLayer::new(128, 256, 4, 0.1);
let enhanced = layer.forward(&query_embedding, &neighbor_embeddings, &edge_weights);
```

## Features

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **Vector Search** | Find similar items in <0.5ms | Fast enough for real-time apps |
| **Cypher Queries** | `MATCH`, `WHERE`, `CREATE`, `RETURN` | Use familiar Neo4j syntax |
| **GNN Layers** | Neural network on the index graph | Search improves with usage |
| **Hyperedges** | Connect 3+ nodes at once | Model complex relationships |
| **Auto-Compression** | 2-32x memory reduction | Store more, pay less |
| **Browser Support** | Full WASM build | Run AI search client-side |

## Performance

```
┌─────────────────────────────────────────────────────────┐
│ Benchmark              RuVector    Pinecone    ChromaDB │
├─────────────────────────────────────────────────────────┤
│ Query Latency (p50)    <0.5ms      ~2ms        ~50ms   │
│ Memory (1M vectors)    ~200MB*     ~2GB        ~3GB    │
│ Browser Support        ✅          ❌          ❌      │
│ Graph Queries          ✅          ❌          ❌      │
│ Self-Improving         ✅          ❌          ❌      │
└─────────────────────────────────────────────────────────┘
* With PQ8 compression enabled
```

## How the GNN Works

Traditional vector search:
```
Query → HNSW Index → Top K Results
```

RuVector with GNN:
```
Query → HNSW Index → GNN Layer → Enhanced Results
                ↑                      │
                └──── learns from ─────┘
```

The GNN layer:
1. Takes your query and its nearest neighbors
2. Applies attention to weigh which neighbors matter
3. Updates representations based on graph structure
4. Returns better-ranked results

Over time, frequently-accessed paths get reinforced, making common queries faster and more accurate.

## Compression Tiers

RuVector automatically compresses cold data:

| Access Frequency | Format | Compression | Example |
|-----------------|--------|-------------|---------|
| Hot (>80%) | f32 | 1x | Active queries |
| Warm (40-80%) | f16 | 2x | Recent docs |
| Cool (10-40%) | PQ8 | 8x | Older content |
| Cold (1-10%) | PQ4 | 16x | Archives |
| Archive (<1%) | Binary | 32x | Rarely used |

## Use Cases

**RAG (Retrieval-Augmented Generation)**
```javascript
// Find relevant context for LLM
const context = differentiableSearch(questionEmbedding, documentEmbeddings, 5, 0.1);
const prompt = `Context: ${context.map(i => docs[i]).join('\n')}\n\nQuestion: ${question}`;
```

**Recommendation Systems**
```cypher
MATCH (user:User)-[:VIEWED]->(item:Product)
MATCH (item)-[:SIMILAR_TO]->(rec:Product)
WHERE NOT (user)-[:VIEWED]->(rec)
RETURN rec ORDER BY rec.score DESC LIMIT 10
```

**Knowledge Graphs**
```cypher
MATCH (concept:Concept)-[:RELATES_TO*1..3]->(related)
WHERE concept.embedding <-> $query_embedding < 0.5
RETURN related
```

## Installation Options

### NPM (Node.js)
```bash
npm install ruvector-gnn
```

### NPM (Browser/WASM)
```bash
npm install ruvector-gnn-wasm
```

### Cargo (Rust)
```bash
cargo add ruvector-core ruvector-graph ruvector-gnn
```

### From Source
```bash
git clone https://github.com/ruvnet/ruvector.git
cd ruvector
cargo build --release
```

## API Overview

### Core Classes

```javascript
// GNN Layer - enhances embeddings using graph structure
const layer = new RuvectorLayer(inputDim, hiddenDim, heads, dropout);
const output = layer.forward(nodeEmbedding, neighborEmbeddings, edgeWeights);

// Compression - reduce memory 2-32x
const compressor = new TensorCompress();
const compressed = compressor.compress(embedding, accessFrequency);
const restored = compressor.decompress(compressed);

// Search - differentiable k-NN
const { indices, weights } = differentiableSearch(query, candidates, k, temperature);
```

### Cypher Queries

```cypher
-- Create nodes
CREATE (a:Person {name: 'Alice', embedding: [0.1, 0.2, 0.3]})

-- Create relationships
MATCH (a:Person), (b:Person)
WHERE a.name = 'Alice' AND b.name = 'Bob'
CREATE (a)-[:KNOWS {since: 2020}]->(b)

-- Query patterns
MATCH (p:Person)-[:KNOWS*1..2]->(friend)
RETURN p.name, collect(friend.name) as friends

-- Vector similarity (extension)
MATCH (d:Document)
WHERE d.embedding <-> $query < 0.5
RETURN d.title, d.embedding <-> $query as distance
```

## Project Structure

```
ruvector/
├── crates/
│   ├── ruvector-core/       # Vector DB engine
│   ├── ruvector-graph/      # Graph DB + Cypher
│   ├── ruvector-gnn/        # GNN layers + training
│   ├── ruvector-gnn-wasm/   # Browser bindings
│   └── ruvector-gnn-node/   # Node.js bindings
└── docs/                    # Documentation
```

## Documentation

- [Getting Started Guide](./docs/guide/GETTING_STARTED.md)
- [Cypher Reference](./docs/api/CYPHER_REFERENCE.md)
- [GNN Architecture](./docs/gnn-layer-implementation.md)
- [Node.js API](./crates/ruvector-gnn-node/README.md)
- [WASM API](./crates/ruvector-gnn-wasm/README.md)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./docs/development/CONTRIBUTING.md).

```bash
# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace
```

## License

MIT License - free for commercial and personal use.

---

<div align="center">

**Built by [rUv](https://ruv.io)** • [GitHub](https://github.com/ruvnet/ruvector)

*Vector search that gets smarter over time.*

</div>
