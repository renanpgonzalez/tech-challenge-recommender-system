# Infraestrutura como Código (IaC) — MLflow Server na AWS

Esta pasta contém as configurações do **Terraform** para o provisionamento automatizado da infraestrutura de nuvem na AWS que hospeda a Central de Experimentos e o Model Registry do **MLflow**.

---

## 1. Desenho da Arquitetura

A arquitetura provisionada segue padrões recomendados de segurança de borda e alta disponibilidade para servidores de tracking:

```text
[Usuário / Pipeline CI/CD] 
       │
       ▼ (HTTPS / mlflow.recommender.cloud-ip.cc)
 [AWS WAF (Web Application Firewall)]  --> Aplica Rate Limiting (máx. 2000 req/5min)
       │
       ▼
 [AWS CloudFront (CDN)]  --> Aplica Geo-Blocking (Apenas conexões de BR e PT)
       │
       ▼ (Porta 5000 / HTTP - Protegido por Security Group)
 [AWS EC2 (t3.micro)]  --> Roda MLflow Server em container Docker com swap de 2GB
       │
       ▼ (IAM Instance Profile / Sem chaves estáticas)
 [AWS S3 (Private Bucket)]  --> Persistência dos artefatos de treinamento
```

---

## 2. Requisitos Prévios

1. **Credenciais AWS**: Chaves de acesso (`Access Key ID` e `Secret Access Key`) com permissão de administrador no terminal.
2. **Bucket S3 para State**: Um bucket chamado `fiappostech9mletgrupo17-fase02-tfstate-967982352747` (ou correspondente ao seu ID de conta) pré-criado na região `sa-east-1` para armazenar o arquivo de estado `.tfstate` do Terraform.
3. **Zona DNS no ClouDNS**: Um domínio de testes ou subdomínio configurado na ClouDNS (ex: `recommender.cloud-ip.cc`).

---

## 3. Como Rodar e Aplicar (Deploy)

Siga a sequência de comandos dentro deste diretório (`/terraform`):

### 1. Inicialização e Validação
```bash
# Inicializa os provedores e conecta com o S3 remoto
terraform init -reconfigure

# Valida a sintaxe dos arquivos
terraform validate
```

### 2. Passo de Validação SSL (ACM)
Como o CloudFront exige que o certificado SSL esteja emitido antes de sua criação, execute o deploy direcionado para o ACM primeiro:
```bash
terraform apply -target=aws_acm_certificate.mlflow_cert -auto-approve
```
* **Ação no DNS**: O comando acima exibirá dois valores nos outputs do terminal: `acm_validation_cname_name` e `acm_validation_cname_value`. Acesse seu painel do ClouDNS, crie um registro do tipo **CNAME** com esses valores e aguarde 2 minutos para propagação.

### 3. Deploy do Restante dos Recursos
Após o CNAV de validação estar configurado, execute o deploy completo:
```bash
terraform apply -auto-approve
```

---

## 4. Como Testar a Infraestrutura

### 1. Testando Acesso Web
* Abra no navegador a URL personalizada cadastrada (ex: `https://mlflow.recommender.cloud-ip.cc`).
* A interface de tracking do MLflow deverá abrir normalmente com criptografia SSL (cadeado ativo).

### 2. Testando a Proteção do AWS WAF (Rate Limiting)
O WAF bloqueia IPs que façam mais de 2000 requisições a cada 5 minutos. Você pode testar disparando múltiplas chamadas rápidas via terminal (ex: usando `curl` em loop):
```bash
for i in {1..120}; do curl -I -s https://mlflow.recommender.cloud-ip.cc | grep "HTTP"; done
```
* **Resultado esperado**: As primeiras requisições retornarão `HTTP/2 200`. Após a centésima chamada, o WAF bloqueará o seu IP retornando erros `HTTP/2 403 Forbidden`.

### 3. Testando o Geo-Blocking (CloudFront)
A distribuição restringe acessos a conexões fora de **Brasil (BR)** e **Portugal (PT)**.
* **Teste**: Ative uma VPN comercial simulando tráfego dos Estados Unidos (US) ou Alemanha (DE) e tente acessar a URL.
* **Resultado esperado**: O CloudFront impedirá o acesso retornando uma tela de erro de bloqueio geográfico.

---

## 5. Como Destruir (Clean Up)

Ao final do ciclo do projeto acadêmico, para evitar cobranças indesejadas no seu cartão da AWS, você pode destruir todos os recursos criados de forma simples:

```bash
terraform destroy -auto-approve
```
*(Nota: O bucket S3 de artefatos possui a flag `force_destroy = true` configurada, o que significa que o Terraform removerá todos os arquivos salvos do MLflow de forma limpa ao rodar o comando).*
