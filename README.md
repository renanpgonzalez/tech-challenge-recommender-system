# Sistema de Recomendação de Produtos — RetailRocket

Este repositório contém a entrega do trabalho de conclusão da **Fase 02** da pós-graduação **FIAP Pós Tech em Machine Learning Engineering**.

O objetivo do projeto é desenvolver, testar, rastrear e versionar um sistema de recomendação de produtos utilizando dados de comportamento implícito (implicit feedback) de e-commerce da plataforma RetailRocket.

---

## 🔗 Links Oficiais do Projeto

* **Documento de Entrega Detalhado**: [entrega-tech-challenge-grupo17.md](file:///Users/brunoabreu/postech/repos/tech-challenge-recommender-system/entrega-tech-challenge-grupo17.md)

### 1.1. Apresentação do Projeto (Método STAR)

Como exigido pelas diretrizes de avaliação da FIAP, o vídeo explicativo do projeto está disponível no link abaixo:

* **Link do Vídeo**: [Clique aqui para assistir ao vídeo de apresentação](https://example.com) *(TODO)*
* **Estrutura de Apresentação (STAR)**:
  - **Situação (Situation)**: O problema de negócio no e-commerce e o contexto do dataset da RetailRocket.
  - **Tarefa (Task)**: Os requisitos de MLOps, rastreamento de experimentos, reprodutibilidade e conformidade com arquitetura de software limpa.
  - **Ação (Action)**: A modelagem PyTorch, o pipeline reprodutível DVC, os testes de CI com GitHub Actions e o provisionamento com Terraform na AWS.
  - **Resultado (Result)**: Comparação de métricas offline (Hit Rate, Coverage), análise do Model Card e validação do ambiente.

---

## 1. Visão Geral do Projeto

O projeto entrega um fluxo completo de engenharia de machine learning (MLOps) composto por:
* Processamento de dados robusto e padronizado;
* Engenharia de recursos (feature engineering) para interações usuário-item;
* Divisão temporal de dados (train-test split);
* Modelo baseline de popularidade;
* Recomendador de deep learning baseado em PyTorch;
* Avaliação comparativa utilizando métricas de ranking (Precision@K, Recall@K, Hit Rate@K, Catalog Coverage@K);
* Rastreamento de logs e parâmetros com MLflow;
* Controle e versionamento de dados e pipelines com DVC;
* Ambiente de desenvolvimento e execução isolado via Docker;
* Registro de modelos e promoção automática com MLflow Model Registry;
* Provisionamento de infraestrutura AWS com Terraform (IaC);
* Esteiras de CI/CD automatizadas via GitHub Actions.

---

## 2. Problema de Negócio

Uma plataforma de e-commerce deseja melhorar a descoberta de produtos para os seus clientes, recomendando produtos de alto interesse com base no seu comportamento histórico.

Dado que não existem avaliações explícitas (estrelas ou notas), o sistema consome dados de feedback implícito através de interações:
* `view` (visualizações de produtos);
* `addtocart` (adições ao carrinho);
* `transaction` (compras concluídas).

---

## 3. Conjunto de Dados (Dataset)

Utilizou-se o dataset público **RetailRocket**, que mapeia o comportamento de navegação de usuários reais.
O dataset bruto é composto por interações sequenciais com carimbos de data/hora (timestamps).

---

## 4. Estratégia de Ponderação de Eventos

Para traduzir as ações implícitas em um score de afinidade contínuo, aplicou-se a seguinte ponderação:

* **Visualização (`view`)**: Peso `1.0` (indica interesse básico).
* **Adição ao Carrinho (`addtocart`)**: Peso `3.0` (indica intenção de compra).
* **Compra (`transaction`)**: Peso `5.0` (indica decisão final de compra).

As interações repetidas por usuário e item são somadas, gerando o atributo `interaction_score`.

---

## 5. Estrutura do Projeto

A organização de pastas segue os padrões de Clean Code e Modularidade recomendados:

```text
├── .github/workflows/       # Workflows automatizados de CI/CD (GitHub Actions)
├── configs/                 # Configurações do ambiente de desenvolvimento
├── context/                 # PDFs de suporte das disciplinas e do desafio
├── data/                    # Dados (ignorado no Git, versionado no DVC)
│   ├── raw/                 # Dados brutos de entrada
│   ├── interim/             # Dados em processamento intermediário
│   └── processed/           # Dados de treino e teste finais
├── dvc-storage/             # Cache local de armazenamento do DVC
├── models/                  # Artefatos locais de modelos salvos
├── reports/                 # Relatórios de performance e Model Cards
├── scripts/                 # Scripts Python executáveis de orquestração do pipeline
├── src/                     # Código fonte modular empacotado
│   └── recommender/
│       ├── data/            # Carregamento e leitura de dados
│       ├── features/        # Engenharia de recursos
│       ├── training/        # Loops de treinamento de modelos
│       ├── evaluation/      # Métricas e rotinas de teste de ranking
│       ├── tracking/        # Helpers de MLflow e promoção de modelos
│       └── models/          # Classes de modelos (Baseline, PyTorch, Factory)
├── terraform/               # Códigos IaC para provisionamento AWS
├── pyproject.toml           # Configurações de dependências do Poetry
└── docker-compose.yml       # Orquestração local do MLflow e treinamento
```

---

## 6. Principais Tecnologias

* **Python 3.12**: Linguagem base estável.
* **Poetry**: Gerenciamento de dependências rigoroso e do ambiente virtual.
* **PyTorch**: Framework de Deep Learning para a rede neural.
* **DVC (Data Version Control)**: Versionamento de dados e estruturação das fases de ML.
* **MLflow**: Rastreamento de parâmetros, métricas e centralização de Model Registry.
* **Docker & Docker Compose**: Garantia de portabilidade de ambientes.
* **Terraform**: Provisionamento IaC para deploy AWS.
* **Pytest & Ruff**: Testes de código e garantia de estilo limpo.

---

## 7. Instalação Local

1. Instale o Poetry (caso não tenha instalado):
   ```bash
   pip install poetry
   ```
2. Clone o repositório e instale as dependências:
   ```bash
   poetry install
   ```
3. Ative o ambiente virtual:
   ```bash
   poetry shell
   ```

---

## 8. Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com base no arquivo de exemplo `.env.example`:

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

---

## 9. Configuração de Dados

O dataset original da RetailRocket pode ser baixado publicamente no Kaggle:
* **Link para Download**: [Kaggle - Retailrocket eCommerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

Após baixar e descompactar o arquivo `.zip`, coloque o arquivo `events.csv` na seguinte estrutura de diretórios do projeto:

```text
data/raw/retailrocket/events.csv
```

### 9.1. Funcionamento do Cache do DVC
O remote do DVC está configurado para salvar o cache de forma local na pasta `dvc-storage/tech-challenge-recommender-system`. 

Por essa razão, ao realizar um clone limpo, o comando `poetry run dvc pull` informará que não existem dados no remote do Git.

Para recriar o pipeline e construir seu cache DVC local:
1. Baixe o dataset da RetailRocket e salve o arquivo `events.csv` na pasta `data/raw/retailrocket/`.
2. Rode o pipeline para processar e treinar:
   ```bash
   poetry run dvc repro
   ```
3. Execute o push para sincronizar o cache DVC local:
   ```bash
   poetry run dvc push
   ```

---

## 10. Pipeline Reprodutível com DVC

O pipeline é composto por 9 etapas orquestradas pelo `dvc.yaml`:

```text
prepare_events → preprocess → split_data → feature_engineering
→ train_baseline → evaluate_baseline → train_neural
→ evaluate_neural → compare_models
```

Para rodar todo o pipeline sequencialmente:
```bash
poetry run dvc repro
```

Para verificar o status das etapas do pipeline:
```bash
poetry run dvc status
```

---

## 11. Parâmetros do Pipeline

Todos os parâmetros globais estão contidos no arquivo `params.yaml`.

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

---

## 12. Modelos

### 12.1. Baseline de Popularidade
Modelo simples baseado na contagem de interações ponderadas acumuladas do conjunto de treino. Recomenda os itens mais populares do catálogo geral.

Implementado em: [baseline.py](file:///Users/brunoabreu/postech/repos/tech-challenge-recommender-system/src/recommender/models/baseline.py)

### 12.2. Recomendador Neural
Rede neural PyTorch baseada em embeddings que recebe o índice de usuário e de item e estima o score contínuo de afinidade através de camadas densas (MLP).

Implementado em: [neural.py](file:///Users/brunoabreu/postech/repos/tech-challenge-recommender-system/src/recommender/models/neural.py)

### 12.3. Padrões de Projeto (Design Patterns)
Para assegurar a modularidade e facilidade de extensão de código:
* **Classe Abstrata**: [BaseRecommender](file:///Users/brunoabreu/postech/repos/tech-challenge-recommender-system/src/recommender/models/base.py) define o contrato com métodos `fit`, `recommend`, `save` e `load`.
* **Padrão Factory**: [RecommenderFactory](file:///Users/brunoabreu/postech/repos/tech-challenge-recommender-system/src/recommender/models/factory.py) permite instanciar de forma transparente os modelos com base em uma string identificadora (`popularity` ou `neural`).

---

## 13. Estratégia de Avaliação Neural

Para otimizar o tempo de avaliação sem testar todo o catálogo para todos os usuários:
1. O baseline de popularidade seleciona 100 itens candidatos.
2. A rede neural reordena estes 100 candidatos.
3. Avaliam-se as métricas Top-10 de ranking.

---

## 14. Métricas de Avaliação

* **Precision@K**: Proporção de recomendados relevantes.
* **Recall@K**: Proporção de relevantes capturados.
* **Hit Rate@K**: Presença de pelo menos um relevante nas recomendações.
* **Coverage@K**: Diversidade/cobertura do catálogo.

A métrica norteadora principal é o **Hit Rate@10**.

---

## 15. Resultados da Versão V1

| Modelo | Precision@10 | Recall@10 | Hit Rate@10 | Coverage@10 |
| ------ | -----------: | --------: | ----------: | ----------: |
| Baseline de Popularidade | 0.0002266 | 0.002266 | 0.002266 | 0.000110 |
| Reranker Neural | 0.0000400 | 0.000400 | 0.000400 | 0.000264 |

O baseline foi superior em acurácia de relevância imediata (Hit Rate), porém o modelo neural se mostrou melhor na diversificação das indicações (cobertura do catálogo).

---

## 16. Servidor de Rastreamento (MLflow)

O MLflow monitora os logs locais em um banco SQLite (`sqlite:///mlflow.db`).
Para iniciar o servidor web do MLflow local:
```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Em seguida, abra: `http://localhost:5000`

---

## 17. MLflow Model Registry

O recomendador neural é registrado no registry sob o nome `retailrocket-neural-recommender`.
Os modelos que cumprem as exigências mínimas recebem a tag de validação `validation_status=approved` e são promovidos automaticamente para o alias `champion` e estágio `Production`.

---

## 18. Docker e Docker Compose

O projeto conta com imagens multi-stage otimizadas.

1. Construir a imagem:
   ```bash
   docker compose build
   ```
2. Validar variáveis locais:
   ```bash
   docker compose run --rm trainer python scripts/validate_env.py
   ```
3. Iniciar servidor MLflow local:
   ```bash
   docker compose up mlflow
   ```
4. Rodar o pipeline do DVC isolado:
   ```bash
   docker compose run --rm trainer dvc repro
   ```

---

## 19. Infraestrutura de Nuvem (Terraform AWS)

Localizada na pasta `terraform/`, a infraestrutura em nuvem provisiona uma arquitetura completa MLOps:
* **S3 Bucket**: Persistência global de artefatos de experimentos.
* **EC2 Instance (`t3.micro`)**: Hospeda o contêiner Docker do servidor MLflow central.
* **CloudFront CDN**: Entrega HTTPS segura via ACM SSL e restrição geográfica (Whitelists BR e PT).
* **AWS WAF**: Bloqueios adicionais de segurança e proteção de ataques (Rate Limit).

Para iniciar e validar:
```bash
cd terraform
terraform init
terraform validate
terraform plan
```

---

## 20. Esteiras de CI/CD (GitHub Actions)

* **`docker-publish.yml`**: Roda testes automatizados, constrói e publica a imagem Docker no Docker Hub, executando o deploy contínuo via AWS SSM.
* **`coverage-publish.yml`**: Roda os testes, calcula a cobertura de código, gera o crachá e publica o HTML no GitHub Pages.
* **`deploy-infra.yml`**: Executa `apply` ou `destroy` do Terraform.
* **`restart-ec2.yml`**: Reinicia a máquina EC2 associada ao MLflow.

---

## 21. Execução de Testes

Rode a suíte de testes unitários local:
```bash
poetry run pytest
```

Validação de qualidade do código (Linter):
```bash
poetry run ruff check . --fix
poetry run ruff format .
```

---

## 22. Status Atual do Projeto

Fase 2 entregue com sucesso: pipelines versionados e reprodutíveis no DVC, rastreamento ativo no MLflow, conteinerização Docker funcional, arquitetura limpa com padrões de projeto aplicados, validação de estilo sem erros e infraestrutura automatizada pronta.
