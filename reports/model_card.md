# Model Card — RetailRocket Neural Recommender V1

## 1. Model Details

### Model Name

`retailrocket-neural-recommender`

### Version

V1

### Registered Model

MLflow Model Registry:

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

### Model Type

Embedding-based neural recommender with MLP head.

### Framework

PyTorch

### Project

Tech Challenge Fase 02 — Product Recommendation System

### Author

Renan Gonzalez

## 2. Intended Use

This model is intended to recommend products in an e-commerce context based on implicit user behavior.

The model uses historical user-item interactions to estimate the relevance of candidate items for a user.

In V1, the model is used as a neural reranker:

```text
Popularity baseline generates candidates
→ Neural model reranks candidate items
→ Top-K recommendations are evaluated
```

## 3. Out-of-Scope Use

This model is not intended for:

* real-time production recommendation without additional validation;
* cold-start users with no previous interaction history;
* cold-start items not present in training mappings;
* sensitive decision-making;
* pricing decisions;
* inventory allocation;
* personalized offers involving sensitive user attributes.

## 4. Dataset

The model was trained using the RetailRocket e-commerce dataset.

Main file used:

```text
data/raw/retailrocket/events.csv
```

The V1 uses only behavioral event data.

Original event types:

```text
view
addtocart
transaction
```

The project standardizes the dataset into:

```text
user_id
item_id
event_type
timestamp
```

## 5. Data Processing

### 5.1. Event Standardization

RetailRocket columns were mapped to the internal schema:

| Original Column | Internal Column |
| --------------- | --------------- |
| `visitorid`     | `user_id`       |
| `itemid`        | `item_id`       |
| `event`         | `event_type`    |
| `timestamp`     | `timestamp`     |

### 5.2. Event Weighting

Implicit feedback events were weighted according to interaction strength:

| Event Type    | Weight |
| ------------- | -----: |
| `view`        |    1.0 |
| `addtocart`   |    3.0 |
| `transaction` |    5.0 |

### 5.3. Feature Engineering

Interactions were aggregated by user-item pair.

Generated features:

| Feature             | Description                                  |
| ------------------- | -------------------------------------------- |
| `interaction_score` | Sum of weighted event interactions           |
| `interaction_count` | Number of interactions between user and item |
| `last_timestamp`    | Last interaction timestamp                   |
| `user_index`        | Encoded user index for embeddings            |
| `item_index`        | Encoded item index for embeddings            |

### 5.4. Train-Test Split

A chronological user-based split was used.

The last interaction of each eligible user was assigned to the test set. Earlier interactions were kept in the training set.

This approach better simulates a recommendation scenario where past behavior is used to predict future interest.

## 6. Training Data

Approximate processed data volume:

| Dataset Step              |      Rows |
| ------------------------- | --------: |
| Preprocessed interactions | 2,756,101 |
| Train interactions        | 2,350,081 |
| Test interactions         |   406,020 |
| Train feature rows        | 1,930,311 |
| Test feature rows         |   397,600 |

The neural V1 was trained using a sample of 200,000 training rows to keep local experimentation feasible.

## 7. Model Architecture

The model receives a user index and an item index.

Architecture:

```text
user_index → user_embedding
item_index → item_embedding
concat(user_embedding, item_embedding)
→ Linear layer
→ ReLU
→ Linear layer
→ predicted interaction score
```

Training objective:

```text
predict weighted interaction score
```

Loss function:

```text
Mean Squared Error
```

## 8. Training Configuration

V1 configuration:

| Parameter             |  Value |
| --------------------- | -----: |
| `embedding_dim`       |      8 |
| `hidden_dim`          |     16 |
| `learning_rate`       |  0.001 |
| `epochs`              |      2 |
| `batch_size`          |   8192 |
| `sample_size`         | 200000 |
| `validation_fraction` |    0.2 |
| `random_seed`         |     42 |

## 9. Training Results

Neural training result:

```json
{
  "train_loss": 3.472243607711792,
  "validation_loss": 3.6423588325500487,
  "epochs_trained": 2,
  "best_epoch": 2
}
```

The loss decreased between epoch 1 and epoch 2, indicating that the model was able to learn from the sampled training data.

## 10. Evaluation Setup

The model was evaluated as a neural reranker.

Evaluation flow:

```text
1. Popularity baseline selects candidate items.
2. Neural model scores candidate user-item pairs.
3. Candidate items are sorted by neural score.
4. Top-K recommendations are evaluated.
```

Evaluation parameters:

| Parameter        | Value |
| ---------------- | ----: |
| `top_k`          |    10 |
| `candidate_size` |   100 |
| `max_users`      | 10000 |

## 11. Metrics

The following ranking metrics were used:

| Metric           | Description                                             |
| ---------------- | ------------------------------------------------------- |
| `precision_at_k` | Fraction of recommended items that are relevant         |
| `recall_at_k`    | Fraction of relevant items retrieved in the top K       |
| `hit_rate_at_k`  | Whether at least one relevant item appears in the top K |
| `coverage_at_k`  | Fraction of catalog items recommended                   |

The primary decision metric for V1 is:

```text
hit_rate_at_k
```

## 12. Model Performance

### 12.1. Popularity Baseline

```json
{
  "precision_at_k": 0.00022660965794768615,
  "recall_at_k": 0.002266096579476861,
  "hit_rate_at_k": 0.002266096579476861,
  "coverage_at_k": 0.00011009384399261931
}
```

### 12.2. Neural Reranker

```json
{
  "precision_at_k": 0.00004,
  "recall_at_k": 0.0004,
  "hit_rate_at_k": 0.0004,
  "coverage_at_k": 0.00026422522558228634
}
```

### 12.3. Comparison

| Model               | Precision@10 | Recall@10 | Hit Rate@10 | Coverage@10 |
| ------------------- | -----------: | --------: | ----------: | ----------: |
| Popularity Baseline |    0.0002266 |  0.002266 |    0.002266 |    0.000110 |
| Neural Reranker     |    0.0000400 |  0.000400 |    0.000400 |    0.000264 |

## 13. Evaluation Interpretation

The popularity baseline outperformed the neural reranker in:

* precision@10;
* recall@10;
* hit_rate@10.

The neural reranker outperformed the baseline in:

* coverage@10.

This means the baseline was better at recommending relevant items in V1, while the neural model produced more diverse recommendations across the catalog.

The selected production-style registered model is the neural recommender because the project requires a PyTorch neural model and MLflow Model Registry lifecycle. However, from a strict relevance perspective, the popularity baseline remains the stronger model in V1.

## 14. Limitations

Main limitations of the V1 model:

* trained on a sample of 200,000 rows instead of the full feature dataset;
* uses a regression objective instead of a ranking-oriented objective;
* does not use negative sampling;
* does not use item metadata;
* does not use category information;
* does not support unknown users or unknown items;
* reranking depends on popularity-generated candidates;
* low absolute relevance metrics;
* not validated in online A/B testing;
* not production-ready for live recommendation traffic.

## 15. Ethical and Fairness Considerations

The model uses behavioral interaction data and does not explicitly use sensitive demographic attributes.

However, recommendation systems can still reinforce popularity bias and exposure bias.

Potential risks:

* over-recommending already popular items;
* underexposing niche items;
* reinforcing historical user behavior;
* limited personalization for new or low-activity users.

Mitigation opportunities:

* monitor catalog coverage;
* introduce diversity-aware reranking;
* include fairness and exposure metrics;
* evaluate long-tail item performance;
* monitor recommendation drift over time.

## 16. Privacy Considerations

The dataset uses anonymized user and item identifiers.

The model does not require personal user attributes.

No personally identifiable information is used in V1.

## 17. Reproducibility

The model can be reproduced using the DVC pipeline.

Main command:

```bash
poetry run dvc repro
```

The project also includes:

* `dvc.yaml`;
* `dvc.lock`;
* `params.yaml`;
* Poetry lock file;
* Dockerfile;
* Docker Compose setup;
* MLflow tracking database;
* tests with pytest.

## 18. MLflow Tracking

The model and experiments were tracked with MLflow.

Tracked information includes:

* parameters;
* metrics;
* artifacts;
* model files;
* training history;
* model registration metadata.

Main tracked runs include:

```text
baseline_popularity
neural_recommender_sample_v1
neural_recommender_sample_v2
neural_recommender_dvc
register_neural_recommender
```

## 19. Model Registry Status

Registered model:

```text
retailrocket-neural-recommender
```

Version:

```text
Version 1
```

Aliases:

```text
staging
champion
```

Tags:

```text
model_type: neural_reranker
validation_status: approved
decision_metric: hit_rate_at_k
```

## 20. Recommended Use in V1

Recommended use:

* demonstration of a PyTorch-based recommender;
* offline recommendation experiment;
* MLOps pipeline validation;
* baseline vs neural comparison;
* foundation for future recommender improvements.

Not recommended use:

* direct live production serving;
* business-critical recommendation decisions;
* cold-start recommendation;
* final personalization engine.

## 21. Future Improvements

Recommended V2 improvements:

1. Train the neural model using a larger sample or full dataset.
2. Add negative sampling.
3. Replace MSE with a ranking-oriented loss.
4. Add item metadata and category features.
5. Add stronger baselines, such as matrix factorization or implicit ALS.
6. Improve candidate generation beyond popularity.
7. Evaluate long-tail recommendations.
8. Add inference API.
9. Add batch scoring pipeline.
10. Use online evaluation or simulated A/B testing.

## 22. Final Decision

For V1, the popularity baseline is the best model in offline relevance metrics.

The neural recommender is retained and registered because it satisfies the PyTorch-based model requirement and demonstrates the full ML lifecycle:

```text
training
evaluation
tracking
versioning
registration
promotion
```

The final technical conclusion is:

The V1 successfully delivers a reproducible recommendation system pipeline. The neural model provides a valid MLOps and PyTorch implementation, while the baseline remains the strongest model in ranking relevance for this initial experiment.
