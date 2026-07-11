# Relatório de Comparação de Modelos

## Objetivo

Comparar o modelo baseline de popularidade com o recomendador neural utilizando métricas de ordenação (ranking) calculadas em K.

## Modelos Comparados

- Baseline: `popularity_baseline`
- Desafiante (Challenger): `neural_reranker`

## Métricas

| Métrica | popularity_baseline | neural_reranker | Diferença Relativa |
|---|---:|---:|---:|
| precision_at_k | 0.0227% | 0.0040% | -82.35% |
| recall_at_k | 0.2266% | 0.0400% | -82.35% |
| hit_rate_at_k | 0.2266% | 0.0400% | -82.35% |
| coverage_at_k | 0.0110% | 0.0264% | +140.00% |

## Decisão

Modelo selecionado: `popularity_baseline`

O baseline de popularidade apresentou melhor desempenho em hit rate, precision e recall neste experimento.

O modelo neural aumentou a cobertura do catálogo (coverage), mas não superou o baseline em relevância de ordenação.

## Interpretação Técnica

O modelo neural foi treinado como um recomendador PyTorch baseado em embeddings e MLP, sendo avaliado por meio do reordenamento de candidatos (reranking).

A versão atual (V1) utiliza uma amostra dos dados de treino e um objetivo simples de regressão sobre o score de interação (MSE).

Melhorias futuras devem explorar amostras maiores de treino, funções de perda voltadas a ranking (como perdas bayesianas ou contrastivas), estratégias robustas de amostragem negativa (negative sampling) e geração de candidatos aprimorada.
