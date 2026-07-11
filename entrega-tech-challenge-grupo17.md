# Entrega - Tech Challenge (Fase 2)

**Curso:** FIAP Pós Tech - Machine Learning Engineering
**Turma:** 9MLET

**Autores (Grupo 17):**
* Bruno Machado Abreu (RM372965)
* Renan Prado Gonzalez (RM374089)
* Davi Coene Rosa (RM371466)
* Paulo Henrique Alves Krempel (RM374144)
* Pedro Gabriel Pereira do Nascimento (RM372994)

---

## 🔗 Links Oficiais do Projeto

1. **Repositório do Código (Github):**
   * [https://github.com/renanpgonzalez/tech-challenge-recommender-system](https://github.com/renanpgonzalez/tech-challenge-recommender-system)
   * *Nota 1:* Todo o código fonte, pipeline de dados versionado (DVC), tracking de experimentos e Model Registry (MLflow), ambiente containerizado (Docker/Docker Compose) e a suíte de 92 testes automatizados encontram-se neste repositório.
   * *Nota 2 (Git Flow e Avaliação de Commits):* A branch `main` possui proteção estrita e só aceita commits oriundos de Pull Requests (PRs). Para manter o histórico limpo, utilizamos a estratégia de **Squash and Merge**. Portanto, para acompanhar a granularidade mais detalhada do trabalho e da evolução do código, solicitamos que os avaliadores analisem as abas de **Pull Requests** (fechados), pois cada commit na branch `main` possui em sua mensagem o link direto para o PR correspondente contendo todos os commits originais.

2. **Apresentação do Projeto (Vídeo STAR):**
   * **Vídeo Link (YouTube/Loom/Drive):** [https://www.youtube.com/watch?v=INSERT_YOUR_STAR_VIDEO_ID_HERE](https://www.youtube.com/watch?v=INSERT_YOUR_STAR_VIDEO_ID_HERE)
   * *Nota:* Vídeo pitch explicativo de 5 minutos detalhando a Situação (Situation), Tarefa (Task), Ações (Action) e Resultados (Result) da solução de recomendação construída.

3. **Deploy em Ambiente de Produção (Central de Experimentos MLflow na AWS):**
   * **MLflow Tracking UI (CloudFront):** [https://mlflow.recommender.cloud-ip.cc](https://mlflow.recommender.cloud-ip.cc)
   * **S3 Artifact Storage:** `fiappostech9mletgrupo17-fase02-mlflow-artifacts`
   * *Nota 1 (Arquitetura de Rastreamento):* A infraestrutura de nuvem da Fase 2 foi estruturada utilizando Terraform (localizada na pasta `terraform/`). Ela cria uma central de rastreamento de experimentos do MLflow em uma instância EC2 (t3.micro) integrada ao S3 para gravação de modelos e artefatos de treinamento remotamente.
   * *Nota 2 (Segurança & WAF):* O servidor está protegido por HTTPS seguro (AWS ACM) via CloudFront CDN. Conta com controle geográfico (Geo-Blocking) restringindo o acesso apenas a IPs do Brasil e de Portugal, além de regras de AWS WAF aplicando Rate Limiting (máximo de 100 requisições a cada 5 minutos por IP) na borda da rede.
