# Tech Challenge Fase 02 — Product Recommendation System

## 1. Project Overview

This project implements a product recommendation system for an e-commerce scenario using user-item interaction data.

The solution was built as a Machine Learning Engineering pipeline, covering:

* data ingestion;
* preprocessing;
* feature engineering;
* baseline recommendation model;
* neural recommender with PyTorch;
* recommendation metrics;
* experiment tracking with MLflow;
* data and pipeline versioning with DVC;
* Docker-based execution environment;
* MLflow Model Registry.

The project uses the RetailRocket e-commerce dataset and focuses on recommending products based on user behavior events such as product views, cart additions and transactions.

### 1.1. Presentation Video (STAR Method)

As required by the Phase 2 Capstone/Tech Challenge evaluation guidelines, the project explanation and demonstration video (maximum 5 minutes) is available at:

* **Video Link**: [Click here to watch the presentation video](https://example.com) *(Please replace this with your recorded Loom/YouTube/Drive link)*
* **Presentation Structure**: The video follows the STAR methodology:
  - **Situation**: Business problem and RetailRocket dataset context.
  - **Task**: MLOps, reproducibility, and architectural requirements.
  - **Action**: Design patterns, DVC pipelines, MLflow logging, and multi-stage Docker.
  - **Result**: Ranking metrics comparisons, Model Card review, and test validations.

## 2. Business Problem

An e-commerce company wants to improve product discovery by recommending relevant items to users based on previous navigation and interaction behavior.

The recommendation task is based on implicit feedback. Instead of explicit ratings, the system uses behavioral events:

* `view`;
* `addtocart`;
* `transaction`.

These events are transformed into weighted interaction scores and used to train recommendation models.

## 3. Dataset

The project uses the RetailRocket e-commerce dataset.

Main file used in V1:

```text
data/raw/retailrocket/events.csv
```

Original columns:

```text
timestamp
visitorid
event
itemid
transactionid
```

Internal standardized schema:

```text
timestamp
user_id
event_type
item_id
```

Only `events.csv` is used in the V1 pipeline. Other RetailRocket files, such as `category_tree.csv` and `item_properties`, were kept as future improvement opportunities.

### Processed Volume

After preprocessing and feature engineering, the V1 pipeline produced approximately:

| Step                      |      Rows |
| ------------------------- | --------: |
| Preprocessed interactions | 2,756,101 |
| Train interactions        | 2,350,081 |
| Test interactions         |   406,020 |
| Train feature rows        | 1,930,311 |
| Test feature rows         |   397,600 |

## 4. Event Weighting Strategy

The dataset contains implicit feedback events. To represent different levels of user intent, each event type receives a weight:

| Event Type    | Weight |
| ------------- | -----: |
| `view`        |    1.0 |
| `addtocart`   |    3.0 |
| `transaction` |    5.0 |

The weighted score is aggregated by `user_id` and `item_id` during feature engineering.

## 5. Project Structure

```text
tech-challenge-recommender-system/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── models/
│   ├── baseline/
│   └── neural/
├── reports/
│   ├── figures/
│   └── model_card.md
├── scripts/
├── src/
│   └── recommender/
│       ├── config/
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       ├── tracking/
│       └── training/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── pyproject.toml
├── poetry.lock
├── .env.example
├── .gitignore
├── .dockerignore
└── .pre-commit-config.yaml
```

## 6. Main Technologies

| Area                         | Tools                     |
| ---------------------------- | ------------------------- |
| Language                     | Python 3.12               |
| Dependency management        | Poetry                    |
| Data processing              | pandas, pyarrow           |
| Machine Learning             | PyTorch, Scikit-Learn     |
| Experiment tracking          | MLflow                    |
| Data and pipeline versioning | DVC                       |
| Testing                      | pytest                    |
| Code quality                 | Ruff, pre-commit          |
| Containerization             | Docker, Docker Compose    |
| Configuration                | Pydantic Settings, `.env` |

## 7. Installation

### 7.1. Clone the repository

```bash
git clone <repository-url>
cd tech-challenge-recommender-system
```

### 7.2. Install dependencies

```bash
poetry install
```

### 7.3. Validate the environment

```bash
poetry run python scripts/validate_env.py
```

## 8. Environment Variables

Create a `.env` file based on `.env.example`.

Example:

```env
APP_ENV=development
RANDOM_SEED=42
DATA_RAW_DIR=data/raw
DATA_INTERIM_DIR=data/interim
DATA_PROCESSED_DIR=data/processed
MODEL_DIR=models
REPORT_DIR=reports
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=product-recommender
```

## 9. Data Setup

The raw RetailRocket events dataset should be available at:

```text
data/raw/retailrocket/events.csv
```

By default, the DVC remote is configured to save cached files locally in the workspace folder `dvc-storage/tech-challenge-recommender-system` (as defined in `.dvc/config`).

Because this remote points to a local directory, cloning the repository to a new machine means `poetry run dvc pull` will report missing cache files. 

To set up the dataset and generate the DVC cache locally:
1. Download the RetailRocket dataset and place `events.csv` (along with properties if needed) inside the `data/raw/retailrocket/` folder.
2. Run the DVC pipeline to process the data, train the models, and run evaluations:
   ```bash
   poetry run dvc repro
   ```
3. Sometime after the run completes successfully, push the newly generated files to your local `dvc-storage` remote to keep the cache in sync:
   ```bash
   poetry run dvc push
   ```

## 10. Reproducible Pipeline with DVC

The project uses DVC to define and reproduce the complete ML pipeline.

Pipeline stages:

```text
prepare_events
preprocess
split
features_train
features_test
baseline_experiment
neural_experiment
neural_evaluation
compare_models
```

To reproduce the full pipeline:

```bash
poetry run dvc repro
```

To check pipeline status:

```bash
poetry run dvc status
```

To push DVC artifacts to the configured remote:

```bash
poetry run dvc push
```

## 11. Pipeline Parameters

Main parameters are stored in `params.yaml`.

Example:

```yaml
project:
  random_seed: 42

mlflow:
  tracking_uri: sqlite:///mlflow.db
  experiment_name: product-recommender

evaluation:
  top_k: 10
  candidate_size: 100
  max_users: 10000

neural:
  embedding_dim: 8
  hidden_dim: 16
  learning_rate: 0.001
  epochs: 2
  batch_size: 8192
  sample_size: 200000
```

## 12. Models

### 12.1. Popularity Baseline

The baseline model recommends the most popular items based on the training data.

This model is simple, fast and useful as a reference point.

Implemented in:

```text
src/recommender/models/baseline.py
```

Main script:

```bash
poetry run python scripts/run_baseline_experiment.py \
  --train-path data/processed/train_features.parquet \
  --test-path data/processed/test_features.parquet \
  --model-output-path models/baseline/popularity_model.json \
  --metrics-output-path reports/baseline_metrics.json \
  --top-k 10 \
  --tracking-uri sqlite:///mlflow.db \
  --experiment-name product-recommender
```

### 12.2. Neural Recommender

The neural recommender is an embedding-based PyTorch model.

Architecture:

```text
user_index → user_embedding
item_index → item_embedding
concat(user_embedding, item_embedding)
→ MLP
→ predicted interaction score
```

Implemented in:

```text
src/recommender/models/neural.py
```

Main script:

```bash
poetry run python scripts/run_neural_experiment.py \
  --train-path data/processed/train_features.parquet \
  --model-output-path models/neural/neural_recommender.pt \
  --metrics-output-path reports/neural_train_metrics.json \
  --history-output-path reports/neural_train_history.json \
  --epochs 2 \
  --batch-size 8192 \
  --embedding-dim 8 \
  --hidden-dim 16 \
  --learning-rate 0.001 \
  --sample-size 200000 \
  --tracking-uri sqlite:///mlflow.db \
  --experiment-name product-recommender \
  --run-name neural_recommender_v1
```

### 12.3. Design Patterns (Clean Code)

To adhere to clean code standards and strict software design principles, the project implements the following design patterns:

* **Abstract Base Class (ABC)**: [BaseRecommender](file:///Users/brunoabreu/postech/repos/tech-challenge-recommender-system/src/recommender/models/base.py) defines the contract for all recommendation engines, ensuring that they consistently implement `fit`, `recommend`, `save`, and `load` methods.
* **Factory Pattern**: [RecommenderFactory](file:///Users/brunoabreu/postech/repos/tech-challenge-recommender-system/src/recommender/models/factory.py) decouples client scripts from concrete model instantiations. It allows creating any supported recommender (e.g. `popularity` or `neural`) using a simple key parameter.

## 13. Neural Evaluation Strategy

The neural model is evaluated using candidate reranking.

Instead of scoring the entire item catalog for every user, the system uses a two-step approach:

```text
Popularity model generates candidate items
→ Neural model reranks the candidates
→ Ranking metrics are calculated
```

This design makes evaluation more efficient and closer to common recommender system architectures.

Evaluation script:

```bash
poetry run python scripts/evaluate_neural.py \
  --train-path data/processed/train_features.parquet \
  --test-path data/processed/test_features.parquet \
  --model-path models/neural/neural_recommender.pt \
  --baseline-model-path models/baseline/popularity_model.json \
  --metrics-output-path reports/neural_eval_metrics.json \
  --top-k 10 \
  --candidate-size 100 \
  --max-users 10000
```

## 14. Evaluation Metrics

The project compares models using ranking metrics at K:

| Metric           | Description                                             |
| ---------------- | ------------------------------------------------------- |
| `precision_at_k` | Fraction of recommended items that are relevant         |
| `recall_at_k`    | Fraction of relevant items retrieved in the top K       |
| `hit_rate_at_k`  | Whether at least one relevant item appears in the top K |
| `coverage_at_k`  | Catalog coverage across recommendations                 |

The main decision metric in V1 is `hit_rate_at_k`.

## 15. V1 Results

### 15.1. Baseline Popularity Model

```json
{
  "precision_at_k": 0.00022660965794768615,
  "recall_at_k": 0.002266096579476861,
  "hit_rate_at_k": 0.002266096579476861,
  "coverage_at_k": 0.00011009384399261931
}
```

### 15.2. Neural Reranker

```json
{
  "precision_at_k": 0.00004,
  "recall_at_k": 0.0004,
  "hit_rate_at_k": 0.0004,
  "coverage_at_k": 0.00026422522558228634
}
```

### 15.3. Model Comparison

| Model               | Precision@10 | Recall@10 | Hit Rate@10 | Coverage@10 |
| ------------------- | -----------: | --------: | ----------: | ----------: |
| Popularity Baseline |    0.0002266 |  0.002266 |    0.002266 |    0.000110 |
| Neural Reranker     |    0.0000400 |  0.000400 |    0.000400 |    0.000264 |

### 15.4. Interpretation

The popularity baseline outperformed the neural reranker in precision, recall and hit rate.

The neural model achieved higher catalog coverage, meaning it recommended a broader set of items.

The V1 conclusion is that the popularity baseline is the best-performing model for relevance, while the neural model demonstrates a complete PyTorch-based recommender architecture integrated with the MLOps pipeline.

Future improvements should focus on:

* larger neural training samples;
* ranking-oriented loss functions;
* negative sampling;
* stronger candidate generation;
* item metadata features;
* category-aware recommendations.

## 16. MLflow Tracking

MLflow is used to track:

* parameters;
* metrics;
* artifacts;
* model files;
* training history;
* model registration.

Tracking URI:

```text
sqlite:///mlflow.db
```

To open the MLflow UI:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://localhost:5000
```

Tracked runs include:

```text
baseline_popularity
neural_recommender_sample_v1
neural_recommender_sample_v2
neural_recommender_dvc
register_neural_recommender
```

## 17. MLflow Model Registry

The neural recommender was registered in MLflow Model Registry as:

```text
retailrocket-neural-recommender
```

Registered version:

```text
Version 1
```

Aliases:

```text
staging
champion
```

Model version tags:

```text
model_type: neural_reranker
validation_status: approved
decision_metric: hit_rate_at_k
```

Registration script:

```bash
poetry run python scripts/register_model.py \
  --model-path models/neural/neural_recommender.pt \
  --metrics-path reports/neural_eval_metrics.json \
  --registered-model-name retailrocket-neural-recommender \
  --tracking-uri sqlite:///mlflow.db \
  --experiment-name product-recommender \
  --run-name register_neural_recommender
```

## 18. Docker

The project includes a multi-stage Dockerfile and Docker Compose setup.

Build the image:

```bash
docker compose build
```

Validate the container environment:

```bash
docker compose run --rm trainer python scripts/validate_env.py
```

Run MLflow with Docker Compose:

```bash
docker compose up mlflow
```

Run the DVC pipeline inside Docker:

```bash
docker compose run --rm trainer dvc repro
```

## 19. Cloud Infrastructure with Terraform (AWS)

The project includes an enterprise-grade cloud deployment architecture located in the `terraform/` directory.

### 19.1. Components
* **AWS S3**: Bucket `fiappostech9mletgrupo17-fase02-mlflow-artifacts` for storing model artifacts centrally.
* **AWS EC2**: Instance `t3.micro` hosting the MLflow server via Docker, with an automated bootstrap script (`user_data`) configuring 2GB swap and running the tracking server.
* **AWS CloudFront**: CDN mapping to the EC2 instance origin on port 5000, serving HTTPS and applying geographical whitelisting (restricting traffic to IPs of BR and PT).
* **AWS ACM**: Free SSL certificate validation via DNS for `mlflow.recommender.cloud-ip.cc`.
* **AWS WAF**: Web Application Firewall restricting rate limits to 100 requests per 5 minutes per IP.

### 19.2. How to Initialize
1. Initialize Terraform provider:
   ```bash
   cd terraform
   terraform init
   ```
2. Validate syntax:
   ```bash
   terraform validate
   ```
3. Plan resources:
   ```bash
   terraform plan
   ```

## 20. CI/CD Pipelines (GitHub Actions)

The repository integrates automated workflows under `.github/workflows/`:

* **`docker-publish.yml`**: Runs Pytest and Ruff lint checks on every commit, builds the custom recommender Docker image, pushes it to Docker Hub, and triggers a rolling update on the EC2 instance via AWS SSM.
* **`coverage-publish.yml`**: Executes unit tests, calculates test coverage, generates a coverage badge, and publishes an HTML coverage report directly to GitHub Pages.
* **`deploy-infra.yml`**: Triggers manual workflow actions to apply or destroy Terraform infrastructure.
* **`restart-ec2.yml`**: Triggers manual workflow actions to reboot the MLflow EC2 server.

## 21. Testing

Run all tests:

```bash
poetry run pytest
```

Run code quality checks:

```bash
poetry run ruff check . --fix
poetry run ruff format .
poetry run pre-commit run --all-files
```

## 22. Current Project Status

V1 completed:

* project structure;
* Poetry environment;
* Ruff and pre-commit;
* pytest test suite;
* RetailRocket dataset adapter;
* preprocessing pipeline;
* temporal train-test split;
* feature engineering;
* popularity baseline;
* PyTorch neural recommender;
* baseline evaluation;
* neural evaluation;
* model comparison report;
* MLflow tracking;
* DVC data versioning;
* DVC reproducible pipeline;
* Docker multi-stage environment;
* Docker Compose services;
* MLflow Model Registry;
* registered neural model with staging and champion aliases.

## 23. Limitations

The V1 neural model did not outperform the popularity baseline in relevance metrics.

Main limitations:

* neural model trained on a sample of 200,000 rows;
* simple regression objective over interaction score;
* no negative sampling strategy;
* no item metadata used;
* no category or product attributes used;
* candidate generation based only on popularity;
* evaluation limited to known users and known items.

## 24. Future Improvements

Recommended improvements for V2:

* implement negative sampling;
* use ranking loss instead of MSE;
* include item metadata and categories;
* train with larger data samples or full dataset;
* experiment with matrix factorization baselines;
* add LightFM or implicit ALS as stronger baselines;
* improve candidate generation;
* add inference API;
* deploy as a service;
* automate model selection based on evaluation metrics.

## 25. Semantic Commit History

The project was built using semantic commits, including:

```text
feat: add feature engineering pipeline
feat: add popularity baseline recommender
feat: add recommendation evaluation metrics
feat: add train test split pipeline
feat: add mlflow tracking for baseline
feat: add pytorch neural recommender components
feat: add neural training pipeline
feat: add reusable feature mappings
feat: add retailrocket dataset adapter
perf: optimize popularity baseline recommendations
feat: add mlflow tracking for neural training
perf: add neural training sampling and progress logs
feat: add neural recommendation evaluation
feat: add model comparison report
chore: initialize dvc data versioning
feat: add dvc reproducible pipeline
feat: add dockerized training environment
feat: add mlflow model registry promotion
```

## 26. Final V1 Conclusion

The V1 successfully delivers a complete Machine Learning Engineering workflow for a product recommendation system.

The popularity baseline is the best-performing model in relevance metrics, while the neural recommender demonstrates the required PyTorch architecture and full MLOps integration.

The project is reproducible, tested, tracked with MLflow, versioned with DVC, containerized with Docker and includes model lifecycle management through MLflow Model Registry.
