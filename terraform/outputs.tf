output "acm_validation_cname_name" {
  description = "NOME do Registro CNAME a ser criado no ClouDNS para a AWS liberar o certificado"
  value       = tolist(aws_acm_certificate.mlflow_cert.domain_validation_options)[0].resource_record_name
}

output "acm_validation_cname_value" {
  description = "VALOR do Registro CNAME a ser colado no ClouDNS"
  value       = tolist(aws_acm_certificate.mlflow_cert.domain_validation_options)[0].resource_record_value
}

output "cloudfront_secure_url_aws" {
  description = "Acesso pela URL nativa da AWS (Garantia de que o CloudFront subiu)"
  value       = "https://${aws_cloudfront_distribution.mlflow_cdn.domain_name}"
}

output "mlflow_custom_domain" {
  description = "Acesso final oficial da Banca (Seu domínio ClouDNS)"
  value       = "https://${var.mlflow_domain}"
}

output "s3_bucket_name" {
  description = "Nome do bucket S3 criado para artefatos do MLflow"
  value       = aws_s3_bucket.mlflow_artifacts.id
}
