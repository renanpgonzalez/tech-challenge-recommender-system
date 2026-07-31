# Sistema de Recomendação de Produtos — RetailRocket

Este repositório contém a entrega do trabalho de conclusão da **Fase 02** da pós-graduação **FIAP Pós Tech em Machine Learning Engineering**.

O objetivo do projeto é desenvolver, testar, rastrear e versionar um sistema de recomendação de produtos utilizando dados de comportamento implícito (implicit feedback) de e-commerce da plataforma RetailRocket.

---

## 🔗 Links Oficiais do Projeto

* **Documento de Entrega Detalhado**: [entrega-tech-challenge-grupo17.md](entrega-tech-challenge-grupo17.md)
* **Repositório GitHub**: [renanpgonzalez/tech-challenge-recommender-system](https://github.com/renanpgonzalez/tech-challenge-recommender-system)
* **Vídeo de Apresentação (STAR)**: [https://www.youtube.com/watch?v=KXa8MuEMq4E](https://www.youtube.com/watch?v=KXa8MuEMq4E)
* **MLflow Tracking UI (Produção AWS)**: [https://mlflow.recommender.cloud-ip.cc](https://mlflow.recommender.cloud-ip.cc)
* **Imagem Docker Hub**: [techchallengefase02/recommender-system](https://hub.docker.com/r/techchallengefase02/recommender-system)

---

## ⚡ Quick Start — Para a Banca Avaliadora

Forma mais rápida de rodar o projeto localmente, sem AWS, sem GPU:

```bash
# 1. Clonar o repositório
git clone https://github.com/renanpgonzalez/tech-challenge-recommender-system.git
cd tech-challenge-recommender-system

# 2. Copiar variáveis de ambiente
cp .env.example .env

# 3. Subir o servidor MLflow via Docker Compose
docker compose up mlflow
# Acesse: http://localhost:5000
```

> O Docker Compose usa armazenamento **local** (sem necessidade de AWS ou credenciais).

---

## 1. Pré-requisitos

| Ferramenta | Versão Mínima | Como instalar |
|---|---|---|
| **Python** | 3.12 | [python.org](https://www.python.org/downloads/) |
| **Poetry** | 2.0+ | `pip install poetry` |
| **Docker** | 24+ | [docs.docker.com](https://docs.docker.com/get-docker/) |
| **Docker Compose** | v2 | Incluído no Docker Desktop |
| **Git** | 2.x | [git-scm.com](https://git-scm.com/) |

> **Nota**: GPU não é necessária. O projeto usa PyTorch CPU-only (`torch 2.12.0`) para garantir portabilidade máxima.

---

## 2. Visão Geral do Projeto

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

## 3. Problema de Negócio

Uma plataforma de e-commerce deseja melhorar a descoberta de produtos para os seus clientes, recomendando produtos de alto interesse com base no seu comportamento histórico.

Dado que não existem avaliações explícitas (estrelas ou notas), o sistema consome dados de feedback implícito através de interações:
* `view` (visualizações de produtos);
* `addtocart` (adições ao carrinho);
* `transaction` (compras concluídas).

---

## 4. Conjunto de Dados (Dataset)

Utilizou-se o dataset público **RetailRocket**, que mapeia o comportamento de navegação de usuários reais.
O dataset bruto é composto por interações sequenciais com carimbos de data/hora (timestamps).

---

## 5. Estratégia de Ponderação de Eventos

Para traduzir as ações implícitas em um score de afinidade contínuo, aplicou-se a seguinte ponderação:

* **Visualização (`view`)**: Peso `1.0` (indica interesse básico).
* **Adição ao Carrinho (`addtocart`)**: Peso `3.0` (indica intenção de compra).
* **Compra (`transaction`)**: Peso `5.0` (indica decisão final de compra).

As interações repetidas por usuário e item são somadas, gerando o atributo `interaction_score`.

---

## 6. Estrutura do Projeto

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
├── .env.example             # Template de variáveis de ambiente
├── docker-compose.yml       # Orquestração local do MLflow e treinamento
├── Dockerfile               # Imagem multi-stage otimizada (~865MB, CPU-only)
└── pyproject.toml           # Configurações de dependências do Poetry
```

---

## 7. Principais Tecnologias

* **Python 3.12**: Linguagem base estável.
* **Poetry**: Gerenciamento de dependências rigoroso e do ambiente virtual.
* **PyTorch 2.12 (CPU)**: Framework de Deep Learning para a rede neural.
* **DVC (Data Version Control)**: Versionamento de dados e estruturação das fases de ML.
* **MLflow 3.x**: Rastreamento de parâmetros, métricas e centralização de Model Registry.
* **Docker & Docker Compose**: Garantia de portabilidade de ambientes.
* **Terraform**: Provisionamento IaC para deploy AWS.
* **Pytest & Ruff**: Testes de código e garantia de estilo limpo.

---

## 8. Instalação Local (Poetry)

```bash
# 1. Instale o Poetry (caso não tenha)
pip install poetry

# 2. Clone o repositório
git clone https://github.com/renanpgonzalez/tech-challenge-recommender-system.git
cd tech-challenge-recommender-system

# 3. Instale as dependências
poetry install

# 4. Ative o ambiente virtual
poetry shell
```

> **Nota sobre PyTorch**: O projeto usa `torch 2.12.0` instalado via PyPI (sem CUDA). Isso garante compatibilidade com macOS, Linux e Windows sem necessidade de GPU.

---

## 9. Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com base no arquivo de exemplo:

```bash
cp .env.example .env
```

O `.env.example` está documentado com todos os campos disponíveis:

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

# --- MLflow Server (usado no docker run) ---
# Backend: SQLite local (padrão) ou postgresql://...
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db

# Artefatos: local (padrão) ou s3://bucket/path/
MLFLOW_ARTIFACT_ROOT=mlruns

# CORS: domínio do frontend
MLFLOW_CORS_ORIGINS=http://localhost:5000

# Região AWS (necessário apenas para artefatos em S3)
AWS_DEFAULT_REGION=sa-east-1
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
```

---

## 10. Configuração de Dados

O dataset original da RetailRocket pode ser baixado publicamente no Kaggle:
* **Link para Download**: [Kaggle - Retailrocket eCommerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

Após baixar e descompactar o arquivo `.zip`, coloque o arquivo `events.csv` na seguinte estrutura de diretórios do projeto:

```text
data/raw/retailrocket/events.csv
```

### 10.1. Funcionamento do Cache do DVC
O remote padrão do DVC está configurado para o S3 (usado pela pipeline CI/CD). Para execução local sem credenciais AWS, o `dvc repro` funciona diretamente — o remote só é necessário para cache (`dvc pull`/`dvc push`).

Para recriar o pipeline localmente (sem AWS):
1. Baixe o dataset da RetailRocket e salve o arquivo `events.csv` na pasta `data/raw/retailrocket/`.
2. Rode o pipeline para processar e treinar:
   ```bash
   poetry run dvc repro
   ```

> **Nota para avaliadores**: O passo `dvc push` requer credenciais AWS e é exclusivo da pipeline CI/CD. Avaliadores sem acesso AWS podem ignorá-lo — o `dvc repro` é autossuficiente.

---

## 11. Pipeline Reprodutível com DVC

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

## 12. Parâmetros do Pipeline

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

## 13. Modelos

### 13.1. Baseline de Popularidade
Modelo simples baseado na contagem de interações ponderadas acumuladas do conjunto de treino. Recomenda os itens mais populares do catálogo geral.

Implementado em: [baseline.py](src/recommender/models/baseline.py)

### 13.2. Recomendador Neural
Rede neural PyTorch baseada em embeddings que recebe o índice de usuário e de item e estima o score contínuo de afinidade através de camadas densas (MLP).

Implementado em: [neural.py](src/recommender/models/neural.py)

### 13.3. Padrões de Projeto (Design Patterns)
Para assegurar a modularidade e facilidade de extensão de código:
* **Classe Abstrata**: [BaseRecommender](src/recommender/models/base.py) define o contrato com métodos `fit`, `recommend`, `save` e `load`.
* **Padrão Factory**: [RecommenderFactory](src/recommender/models/factory.py) permite instanciar de forma transparente os modelos com base em uma string identificadora (`popularity` ou `neural`).

---

## 14. Estratégia de Avaliação Neural

Para otimizar o tempo de avaliação sem testar todo o catálogo para todos os usuários:
1. O baseline de popularidade seleciona 100 itens candidatos.
2. A rede neural reordena estes 100 candidatos.
3. Avaliam-se as métricas Top-10 de ranking.

---

## 15. Métricas de Avaliação

* **Precision@K**: Proporção de recomendados relevantes.
* **Recall@K**: Proporção de relevantes capturados.
* **Hit Rate@K**: Presença de pelo menos um relevante nas recomendações.
* **Coverage@K**: Diversidade/cobertura do catálogo.

A métrica norteadora principal é o **Hit Rate@10**.

---

## 16. Resultados da Versão V1

| Modelo | Precision@10 | Recall@10 | Hit Rate@10 | Coverage@10 |
| ------ | -----------: | --------: | ----------: | ----------: |
| Baseline de Popularidade | 0.0002266 | 0.002266 | 0.002266 | 0.000110 |
| Reranker Neural | 0.0000400 | 0.000400 | 0.000400 | 0.000264 |

O baseline foi superior em acurácia de relevância imediata (Hit Rate), porém o modelo neural se mostrou melhor na diversificação das indicações (cobertura do catálogo).

---

## 17. Docker e Docker Compose

O projeto conta com uma imagem Docker multi-stage otimizada (~865MB, PyTorch CPU-only).

### 17.1. Via Docker Compose (recomendado para uso local)

O Docker Compose utiliza armazenamento **local** (SQLite + pasta `mlartifacts/`), sem necessidade de AWS:

```bash
# Subir servidor MLflow local
docker compose up mlflow
# Acesse: http://localhost:5000

# Rodar pipeline DVC dentro do container
docker compose run --rm trainer dvc repro

# Construir a imagem localmente
docker compose build
```

### 17.2. Via `docker run` com `.env` (controle manual)

```bash
# 1. Configure o .env
cp .env.example .env
# Edite MLFLOW_ARTIFACT_ROOT, MLFLOW_BACKEND_STORE_URI conforme necessário

# 2. Rodar com artefatos locais (padrão do .env.example, sem AWS)
docker run -p 5000:5000 --env-file .env \
  techchallengefase02/recommender-system:latest
# Acesse: http://localhost:5000

# 3. Rodar com artefatos em S3 (edite o .env com suas credenciais AWS)
# Altere no .env:
#   MLFLOW_ARTIFACT_ROOT=s3://seu-bucket/
#   AWS_ACCESS_KEY_ID=xxx
#   AWS_SECRET_ACCESS_KEY=xxx
docker run -p 5000:5000 --env-file .env \
  techchallengefase02/recommender-system:latest
```

### 17.3. Variáveis de ambiente do container

| Variável | Padrão | Descrição |
|---|---|---|
| `MLFLOW_BACKEND_STORE_URI` | `sqlite:///mlflow.db` | Backend de metadados |
| `MLFLOW_ARTIFACT_ROOT` | `mlruns` | Destino dos artefatos (local ou S3) |
| `MLFLOW_CORS_ORIGINS` | `http://localhost:5000` | Origens permitidas pela UI |
| `AWS_DEFAULT_REGION` | `sa-east-1` | Região AWS (apenas para S3) |

---

## 18. Servidor de Rastreamento (MLflow)

### Localmente via Poetry:
```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
# Acesse: http://localhost:5000
```

### Localmente via Docker Compose:
```bash
docker compose up mlflow
# Acesse: http://localhost:5000
```

### Em Produção (AWS):
O servidor MLflow de produção está disponível em:
**[https://mlflow.recommender.cloud-ip.cc](https://mlflow.recommender.cloud-ip.cc)**

> Acesso restrito a IPs do Brasil e Portugal (AWS WAF Geo-Blocking).

---

## 19. MLflow Model Registry

O recomendador neural é registrado no registry sob o nome `retailrocket-neural-recommender`.
Os modelos que cumprem as exigências mínimas recebem a tag de validação `validation_status=approved` e são promovidos automaticamente para o alias `champion` e estágio `Production`.

---

## 20. Infraestrutura de Nuvem (Terraform AWS)

Localizada na pasta `terraform/`, a infraestrutura em nuvem provisiona uma arquitetura completa MLOps:
* **S3 Bucket**: Persistência global de artefatos de experimentos.
* **EC2 Instance (`t3.micro`)**: Hospeda o contêiner Docker com o servidor MLflow central e o sistema de recomendação.
* **CloudFront CDN**: Entrega HTTPS segura via ACM SSL e restrição geográfica (Whitelists BR e PT).
* **AWS WAF**: Bloqueios adicionais de segurança e proteção de ataques (Rate Limit: 2000 req/5min por IP).

Para iniciar e validar:
```bash
cd terraform
terraform init
terraform validate
terraform plan
```

---

## 21. Esteiras de CI/CD (GitHub Actions)

* **`docker-publish.yml`**: Roda testes automatizados, constrói e publica a imagem Docker no Docker Hub (~865MB), executando o deploy contínuo via AWS SSM na instância EC2.
* **`coverage-publish.yml`**: Roda os testes, calcula a cobertura de código, gera o crachá e publica o HTML no GitHub Pages.
* **`deploy-infra.yml`**: Executa `apply` ou `destroy` do Terraform.
* **`restart-ec2.yml`**: Reinicia a máquina EC2 associada ao MLflow.

---

## 22. Execução de Testes

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

## 23. Apresentação do Projeto (Método STAR)

Como exigido pelas diretrizes de avaliação da FIAP, o vídeo explicativo do projeto está disponível no YouTube:

**🎬 [https://www.youtube.com/watch?v=KXa8MuEMq4E](https://www.youtube.com/watch?v=KXa8MuEMq4E)**

* **Estrutura de Apresentação (STAR)**:
  - **Situação (Situation)**: O problema de negócio no e-commerce e o contexto do dataset da RetailRocket.
  - **Tarefa (Task)**: Os requisitos de MLOps, rastreamento de experimentos, reprodutibilidade e conformidade com arquitetura de software limpa.
  - **Ação (Action)**: A modelagem PyTorch, o pipeline reprodutível DVC, os testes de CI com GitHub Actions e o provisionamento com Terraform na AWS.
  - **Resultado (Result)**: Comparação de métricas offline (Hit Rate, Coverage), análise do Model Card e validação do ambiente.

---

## 24. Status Atual do Projeto

Fase 2 entregue com sucesso:
- ✅ Pipelines versionados e reprodutíveis no DVC
- ✅ Rastreamento ativo no MLflow (local e produção AWS)
- ✅ Conteinerização Docker funcional (~865MB, CPU-only)
- ✅ Arquitetura limpa com padrões de projeto aplicados
- ✅ 92 testes automatizados com cobertura publicada no GitHub Pages
- ✅ Validação de estilo sem erros (Ruff)
- ✅ Infraestrutura AWS automatizada (Terraform + CI/CD GitHub Actions)
- ✅ MLflow em produção: [https://mlflow.recommender.cloud-ip.cc](https://mlflow.recommender.cloud-ip.cc)
