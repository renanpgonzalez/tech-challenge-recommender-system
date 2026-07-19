variable "aws_region" {
  description = "Região da AWS para deploy da infraestrutura"
  type        = string
  default     = "sa-east-1"
}

variable "instance_type" {
  description = "Tipo da instância EC2 para o servidor do MLflow"
  type        = string
  default     = "t3.micro"
}

variable "mlflow_domain" {
  description = "Domínio DNS customizado para o servidor MLflow"
  type        = string
  default     = "mlflow.recommender.cloud-ip.cc"
}

variable "docker_image_mlflow" {
  description = "Imagem Docker do MLflow a ser executada no servidor"
  type        = string
  default     = "ghcr.io/mlflow/mlflow:latest"
}
