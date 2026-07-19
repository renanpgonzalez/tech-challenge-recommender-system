# Model Card — Recomendador Neural RetailRocket V1

## 1. Detalhes do Modelo

### Nome do Modelo

`retailrocket-neural-recommender`

### Versão

V1

### Modelo Registrado

MLflow Model Registry:

```text
retailrocket-neural-recommender
```

Versão registrada:

```text
Version 1
```

Aliases:

```text
staging
champion
```

### Tipo de Modelo

Recomendador neural baseado em embeddings com camadas densas (MLP).

### Framework

PyTorch

### Projeto

Tech Challenge Fase 02 — Sistema de Recomendação de Produtos

### Autores

Grupo 17

## 2. Uso Pretendido

Este modelo destina-se a recomendar produtos em um contexto de e-commerce com base no comportamento implícito do usuário.

O modelo utiliza o histórico de interações usuário-item para estimar a relevância de itens candidatos para um determinado usuário.

Na versão V1, o modelo funciona como um reordenador neural (reranker):

```text
Baseline de popularidade gera os candidatos
→ Modelo neural reordena os itens candidatos
→ Recomendações Top-K finais são avaliadas
```

## 3. Uso Fora de Escopo

Este modelo não é indicado para:

* recomendação em produção em tempo real sem validações adicionais;
* usuários do tipo "cold-start" sem histórico de interações;
* itens "cold-start" não mapeados no conjunto de treino;
* tomada de decisões críticas ou sensíveis;
* definição dinâmica de preços;
* alocação física de estoque;
* ofertas personalizadas que envolvam atributos sensíveis dos usuários.

## 4. Conjunto de Dados

O modelo foi treinado utilizando o conjunto de dados de e-commerce da RetailRocket.

Arquivo principal utilizado:

```text
data/raw/retailrocket/events.csv
```

A versão V1 consome apenas os dados de eventos comportamentais.

Tipos originais de eventos:

```text
view
addtocart
transaction
```

O projeto padroniza essas colunas para a seguinte estrutura:

```text
user_id
item_id
event_type
timestamp
```

## 5. Processamento de Dados

### 5.1. Padronização de Eventos

As colunas do RetailRocket foram mapeadas para o esquema interno:

| Coluna Original | Coluna Interna |
| --------------- | --------------- |
| `visitorid`     | `user_id`       |
| `itemid`        | `item_id`       |
| `event`         | `event_type`    |
| `timestamp`     | `timestamp`     |

### 5.2. Ponderação de Eventos

Eventos de feedback implícito receberam pesos de acordo com sua intensidade:

| Tipo do Evento | Peso |
| -------------- | ---: |
| `view`         |  1.0 |
| `addtocart`    |  3.0 |
| `transaction`  |  5.0 |

### 5.3. Engenharia de Recursos (Feature Engineering)

As interações foram agregadas por par usuário-item.

Atributos gerados:

| Atributo | Descrição |
| -------- | --------- |
| `interaction_score` | Soma ponderada dos eventos de interação |
| `interaction_count` | Quantidade total de interações entre o usuário e o item |
| `last_timestamp`    | Timestamp da última interação registrada |
| `user_index`        | Índice mapeado do usuário para embeddings |
| `item_index`        | Índice mapeado do item para embeddings |

### 5.4. Divisão de Treino e Teste

Foi aplicada uma divisão cronológica baseada no usuário (User-based Temporal Split).

A última interação de cada usuário qualificado foi atribuída ao conjunto de teste. As interações anteriores foram mantidas no conjunto de treino.

Esta abordagem simula de forma realista um cenário de recomendação real, onde comportamentos passados predizem interesses futuros.

## 6. Dados de Treinamento

Volume aproximado dos dados processados:

| Etapa do Dataset | Linhas |
| ---------------- | -----: |
| Interações Pré-processadas | 2.756.101 |
| Interações de Treino | 2.350.081 |
| Interações de Teste | 406.020 |
| Linhas de Atributos de Treino | 1.930.311 |
| Linhas de Atributos de Teste | 397.600 |

Para viabilizar a experimentação local e testes ágeis, a versão neural V1 foi treinada sobre uma amostra de 200.000 linhas de treino.

## 7. Arquitetura do Modelo

O modelo recebe o índice de um usuário e o índice de um item.

Arquitetura:

```text
user_index → user_embedding (embedding de usuário)
item_index → item_embedding (embedding de item)
concat(user_embedding, item_embedding)
→ Camada Linear (Densa)
→ Ativação ReLU
→ Camada Linear de Saída
→ Score de interação predito
```

Objetivo de treino:

```text
predizer o score ponderado de interação
```

Função de perda (Loss):

```text
Erro Quadrático Médio (MSE Loss)
```

## 8. Configuração de Treinamento

Configuração aplicada na V1:

| Parâmetro | Valor |
| --------- | ----: |
| `embedding_dim` | 8 |
| `hidden_dim` | 16 |
| `learning_rate` | 0.001 |
| `epochs` | 2 |
| `batch_size` | 8192 |
| `sample_size` | 200000 |
| `validation_fraction` | 0.2 |
| `random_seed` | 42 |

## 9. Resultados do Treinamento

Perda de treino registrada:

```json
{
  "train_loss": 3.472243607711792,
  "validation_loss": 3.6423588325500487,
  "epochs_trained": 2,
  "best_epoch": 2
}
```

A perda diminuiu de forma consistente entre a época 1 e a época 2, indicando que o recomendador neural conseguiu convergir nos dados amostrados.

## 10. Configuração de Avaliação

O modelo foi avaliado no fluxo de reordenador neural (reranker).

Fluxo:

```text
1. O baseline de popularidade seleciona itens candidatos para o usuário.
2. O modelo neural pontua os pares usuário-item candidatos.
3. Os itens candidatos são ordenados de forma decrescente pelo score predito.
4. As métricas do Top-K são calculadas para os itens resultantes.
```

Parâmetros de avaliação aplicados:

| Parâmetro | Valor |
| --------- | ----: |
| `top_k` | 10 |
| `candidate_size` | 100 |
| `max_users` | 10000 |

## 11. Métricas Utilizadas

As seguintes métricas de ordenação (ranking) foram monitoradas:

| Métrica | Descrição |
| ------- | --------- |
| `precision_at_k` | Proporção de itens recomendados que são realmente relevantes |
| `recall_at_k`    | Proporção de itens relevantes recuperados no Top-K |
| `hit_rate_at_k`  | Indica se pelo menos um item relevante apareceu no Top-K |
| `coverage_at_k`  | Proporção de itens do catálogo recomendados a pelo menos um usuário |

A métrica principal de decisão na V1 é:

```text
hit_rate_at_k
```

## 12. Desempenho do Modelo

### 12.1. Baseline de Popularidade

```json
{
  "precision_at_k": 0.00022660965794768615,
  "recall_at_k": 0.002266096579476861,
  "hit_rate_at_k": 0.002266096579476861,
  "coverage_at_k": 0.00011009384399261931
}
```

### 12.2. Reranker Neural

```json
{
  "precision_at_k": 0.00004,
  "recall_at_k": 0.0004,
  "hit_rate_at_k": 0.0004,
  "coverage_at_k": 0.00026422522558228634
}
```

### 12.3. Comparação Direta

| Modelo | Precision@10 | Recall@10 | Hit Rate@10 | Coverage@10 |
| ------ | -----------: | --------: | ----------: | ----------: |
| Baseline de Popularidade | 0.0002266 | 0.002266 | 0.002266 | 0.000110 |
| Reranker Neural | 0.0000400 | 0.000400 | 0.000400 | 0.000264 |

## 13. Interpretação da Avaliação

O modelo baseline de popularidade superou o recomendador neural em:

* precisão@10;
* revocação (recall)@10;
* hit rate@10.

O modelo neural superou o baseline em:

* cobertura (coverage)@10.

Isso demonstra que, embora o baseline seja mais preciso para recomendar itens de alta relevância geral na versão V1, o recomendador neural foi capaz de diversificar as indicações ao longo do catálogo, reduzindo a concentração excessiva de itens populares.

O recomendador neural foi selecionado e registrado como "champion" para cumprir a exigência metodológica do trabalho (uso de PyTorch e orquestração do Model Registry com MLflow), mas o modelo de popularidade continua sendo uma referência forte neste experimento inicial.

## 14. Limitações Conhecidas

Principais limitações da versão V1:

* treinado em uma amostra de 200.000 linhas devido a restrições de computação local;
* utiliza um objetivo simples de regressão ao invés de perdas otimizadas para ranking;
* não realiza amostragem negativa (negative sampling);
* não incorpora metadados de itens ou usuários;
* não considera categorias ou preços dos produtos;
* não gerencia adequadamente novos usuários ou itens (cold-start);
* o reordenamento depende diretamente da pré-seleção do baseline de popularidade;
* métricas gerais de relevância são baixas;
* não validado em testes online ou A/B.

## 15. Considerações Éticas e de Justiça (Fairness)

O modelo consome dados comportamentais e não utiliza atributos sensíveis ou demográficos.

No entanto, sistemas de recomendação podem reforçar vieses de popularidade e vieses de exposição.

Riscos potenciais:

* super-recomendação de itens já populares;
* sub-exposição de produtos de cauda longa (long-tail);
* reforço de comportamentos passados repetitivos;
* baixa qualidade de personalização para usuários novos.

Mitigações futuras:

* monitorar a cobertura agregada do catálogo;
* introduzir filtros de diversificação no reordenamento;
* incluir métricas de justiça (exposure fairness);
* validar desempenho sobre itens de cauda longa;
* avaliar a variação de recomendações no tempo.

## 16. Considerações de Privacidade

O dataset original utiliza identificadores numéricos anonimizados de usuários e itens.

Nenhuma informação pessoal identificável (PII) é capturada ou utilizada na versão V1.

## 17. Reprodutibilidade

O pipeline completo pode ser reproduzido usando o DVC.

Comando principal:

```bash
poetry run dvc repro
```

Recursos de reprodutibilidade incluídos no projeto:

* arquivos de pipeline `dvc.yaml` e `dvc.lock`;
* parametrização unificada em `params.yaml`;
* controle de dependências via Poetry;
* ambiente isolado Docker e orquestração Docker Compose;
* logs de testes unitários com pytest.

## 18. Rastreamento com MLflow (Tracking)

Todos os logs de experimentos foram salvos e monitorados no MLflow.

Dados monitorados:

* parâmetros de treino e avaliação;
* métricas de performance e perdas por época;
* artefatos e histórico de logs;
* informações do ciclo de registro de modelos.

Execuções registradas (Runs):

```text
baseline_popularity
neural_recommender_sample_v1
neural_recommender_sample_v2
neural_recommender_dvc
register_neural_recommender
```

## 19. Status do Model Registry

Modelo registrado:

```text
retailrocket-neural-recommender
```

Versão:

```text
Version 1
```

Aliases ativos:

```text
staging
champion
```

Tags associadas:

```text
model_type: neural_reranker
validation_status: approved
decision_metric: hit_rate_at_k
```

## 20. Recomendações de Uso na V1

Recomendado para:

* validação da arquitetura de deep learning baseada em PyTorch;
* exploração prática de MLOps e versionamento de dados;
* comparação de modelos de recomendação em modo offline;
* fundação arquitetural para futuras evoluções de modelagem.

Não recomendado para:

* deploy direto em tráfego de produção sem testes online A/B;
* tomadas de decisão críticas de vendas ou negócios.

## 21. Evoluções Futuras (V2)

Melhorias indicadas para a versão V2:

1. Treinar o modelo utilizando uma amostra significativamente maior ou a totalidade dos dados.
2. Adicionar uma estratégia de amostragem negativa (Negative Sampling).
3. Alterar a função de perda (loss) de regressão para ranking (como BPR Loss ou Triplet Loss).
4. Incorporar metadados dos produtos (como categorias e propriedades).
5. Experimentar recomendadores baseados em fatoração de matrizes clássicos.
6. Adicionar LightFM ou implicit ALS as baselines competitivas.
7. Otimizar a etapa de geração de candidatos (retrieval).
8. Desenvolver uma API de inferência em tempo real.
9. Criar um fluxo de pontuação em lote (batch scoring).
10. Validar recomendações em simulações A/B online.

## 22. Decisão Final

Para a versão V1, o modelo baseline de popularidade obteve as melhores métricas offline de relevância.

Contudo, o recomendador neural baseado em PyTorch é mantido como o modelo selecionado ("champion") no repositório de produção do MLflow, pois demonstra todo o ciclo de engenharia de machine learning exigido na especificação:

```text
Treino → Avaliação → Rastreamento de Métricas → Versionamento (DVC) → Registro de Modelos → Promoção e Alias de Produção
```

Conclusão técnica final:

O projeto cumpre com excelência a entrega de um pipeline robusto, versionado, testado e modular para sistemas de recomendação de comércio eletrônico.
