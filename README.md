# Vector Database Optimization for LLM Applications

A working MVP that optimizes Weaviate vector database performance for LLM-based applications by benchmarking different HNSW indexing configurations.

## 🎯 Project Overview

This project demonstrates how to:
1. Generate text embeddings using Google's Gemini API
2. Store and retrieve embeddings using Weaviate vector database
3. Benchmark different indexing configurations (HNSW parameters)
4. Measure and optimize for query latency, recall, and throughput
5. Provide actionable optimization recommendations

## 📋 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Embeddings Generation**: Uses Google Gemini API for high-quality text embeddings
- **Vector Storage**: Weaviate integration with HNSW indexing
- **Comprehensive Benchmarking**: Tests multiple configurations automatically
- **Performance Metrics**: Measures latency, recall@k, and throughput
- **Caching**: Intelligent caching to avoid redundant API calls
- **Progress Tracking**: Visual progress bars for long-running operations
- **Results Export**: Saves detailed metrics to CSV for further analysis

## 🛠️ Prerequisites

1. **Python 3.8+**
2. **Weaviate** running locally on port 8081
3. **Google Gemini API key**

### Starting Weaviate with Docker

```bash
# Create a docker-compose.yml file
cat > docker-compose.yml << EOF
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8081:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      CLUSTER_HOSTNAME: 'node1'
EOF

# Start Weaviate
docker-compose up -d

# Verify it's running
curl http://localhost:8081/v1/meta
```

## 📦 Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Copy the example env file
copy .env.example .env

# Edit .env and add your Gemini API key
# Get your API key from: https://makersuite.google.com/app/apikey
```

Your `.env` file should look like:
```
GEMINI_API_KEY=your_actual_api_key_here
WEAVIATE_URL=http://localhost:8081
```

## 🚀 Usage

### Quick Start

Run the complete optimization pipeline:

```bash
python main.py
```

This will:
1. Connect to Weaviate and Gemini API
2. Generate embeddings for a 50-document corpus
3. Test multiple HNSW configurations
4. Display optimization results
5. Save detailed metrics to `results.csv`

### Individual Modules

**Test Embedding Generation:**
```bash
python embedding_generator.py
```

**Test Weaviate Connection:**
```bash
python weaviate_client.py
```

**Run Custom Benchmark:**
```bash
python benchmark.py
```

### Example Output

```
=== VECTOR DATABASE OPTIMIZATION RESULTS ===

📊 OVERALL STATISTICS
──────────────────────────────────────────────────────────────────────
Total Configurations Tested: 18
Corpus Size: 50 documents
Number of Test Queries: 10

🏆 BEST CONFIGURATION (Balanced Optimization)
──────────────────────────────────────────────────────────────────────
Index Type: HNSW
ef_construction: 128
ef (search): 64
max_connections: 32

⚡ PERFORMANCE METRICS
──────────────────────────────────────────────────────────────────────
Average Query Time: 42.6 ms
Std Dev Query Time: 8.3 ms
Recall@10: 96.8%
Throughput: 23.5 queries/sec
Insert Time: 1.2 sec

💡 OPTIMIZATION RECOMMENDATIONS
──────────────────────────────────────────────────────────────────────
• For low latency: Use lower ef values (32-64)
• For high recall: Use higher ef_construction (256+) and ef (128+)
• For balanced performance: Use the best configuration shown above
```

## 📁 Project Structure

```
MVP/
├── main.py                    # Main pipeline orchestration
├── embedding_generator.py     # Gemini API integration
├── weaviate_client.py        # Weaviate database operations
├── benchmark.py              # Performance benchmarking
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── results.csv             # Benchmark results (generated)
└── embeddings_cache/       # Cached embeddings (generated)
```

## 🔧 Configuration

### HNSW Parameters

The benchmark tests various combinations of:

- **ef_construction** (64, 128, 256): Quality of index construction
  - Higher values = better recall but slower indexing
  
- **ef** (32, 64, 128): Size of search candidate list
  - Higher values = better recall but slower queries
  
- **max_connections** (16, 32, 64): Graph connectivity
  - Higher values = better recall but more memory

### Customizing the Benchmark

Edit `main.py` to customize benchmark parameters:

```python
results_df = benchmark.run_benchmark(
    ef_construction_values=[64, 128, 256],  # Your values
    ef_values=[32, 64, 128],                # Your values
    max_connections_values=[16, 32],        # Your values
    k=10,                                   # Top-k for recall
    num_iterations=3                        # Repetitions per config
)
```

## 📊 Understanding the Metrics

- **Query Latency**: Time to execute a similarity search (lower is better)
- **Recall@K**: Percentage of relevant results in top-K (higher is better)
- **Throughput**: Number of queries processed per second (higher is better)
- **Insert Time**: Time to index the corpus (lower is better)

## 🐛 Troubleshooting

### Weaviate Connection Error
```
❌ Failed to connect to Weaviate at http://localhost:8081
```
**Solution**: Ensure Weaviate is running:
```bash
docker-compose up -d
curl http://localhost:8081/v1/meta
```

### Missing API Key
```
ValueError: GEMINI_API_KEY not found
```
**Solution**: Create `.env` file with your API key:
```bash
GEMINI_API_KEY=your_key_here
```

### Import Errors
```
ModuleNotFoundError: No module named 'weaviate'
```
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔄 Next Steps

Extend this MVP with:

1. **Multi-Database Comparison**: Add FAISS, Chroma, or Pinecone
2. **Advanced Metrics**: Add precision, F1-score, MRR
3. **Real Dataset**: Test with larger, domain-specific corpora
4. **Dynamic Configuration**: Auto-tune parameters based on workload
5. **Web Interface**: Build a dashboard for interactive benchmarking
6. **Production Optimization**: Add monitoring, logging, and alerts

## 📝 License

This is an educational MVP project for demonstration purposes.

## 🤝 Contributing

Feel free to fork, modify, and extend this project for your use case!

## 📧 Support

For issues or questions, please check:
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Google Gemini API Docs](https://ai.google.dev/docs)
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)
