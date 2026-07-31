# 1. Obter a AMI mais recente do Amazon Linux 2023
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }
}

# 1.5 Obter a lista oficial de IPs do CloudFront gerenciada pela AWS
data "aws_ec2_managed_prefix_list" "cloudfront" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

# 2. Security Group: Permitir entrada apenas na porta 5000 para o MLflow e saida livre
resource "aws_security_group" "mlflow_sg" {
  name        = "recommender_mlflow_sg"
  description = "Permitir trafego web para o MLflow Server"

  ingress {
    description     = "MLflow Port (Only from CloudFront)"
    from_port       = 5000
    to_port         = 5000
    protocol        = "tcp"
    prefix_list_ids = [data.aws_ec2_managed_prefix_list.cloudfront.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# 3. IAM Role: Conceder permissão para SSM e acesso ao Bucket S3
data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ec2_mlflow_role" {
  name               = "recommender_ec2_mlflow_role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

# Permissões do SSM para controle remoto
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.ec2_mlflow_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Permissão customizada para o S3 de artefatos
resource "aws_iam_policy" "s3_access_policy" {
  name        = "recommender_s3_artifacts_policy"
  description = "Acesso de leitura/escrita no bucket S3 do MLflow"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          "arn:aws:s3:::fiappostech9mletgrupo17-fase02-mlflow-artifacts",
          "arn:aws:s3:::fiappostech9mletgrupo17-fase02-mlflow-artifacts/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "s3_access_attach" {
  role       = aws_iam_role.ec2_mlflow_role.name
  policy_arn = aws_iam_policy.s3_access_policy.arn
}

resource "aws_iam_instance_profile" "ec2_mlflow_profile" {
  name = "recommender_ec2_mlflow_profile"
  role = aws_iam_role.ec2_mlflow_role.name
}

# 4. Instância EC2 (t3.micro - Free Tier) que rodará o Docker
resource "aws_instance" "mlflow_server" {
  ami                  = data.aws_ami.amazon_linux_2023.id
  instance_type        = var.instance_type
  iam_instance_profile = aws_iam_instance_profile.ec2_mlflow_profile.name

  vpc_security_group_ids = [aws_security_group.mlflow_sg.id]

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required" # IMDSv2
    http_put_response_hop_limit = 2          # Permite que o Docker acesse o IMDS
  }

  user_data = <<-EOF
              #!/bin/bash
              
              # 1. Cria 2GB de Swap para proteger a máquina t3.micro (1GB RAM) contra OOM Killer
              dd if=/dev/zero of=/swapfile bs=128M count=16
              chmod 600 /swapfile
              mkswap /swapfile
              swapon /swapfile
              echo "/swapfile swap swap defaults 0 0" >> /etc/fstab

              # 2. Atualiza os pacotes e instala o Docker
              dnf update -y
              dnf install -y docker
              
              # Inicia o serviço do Docker
              systemctl start docker
              systemctl enable docker
              
              # Adiciona o usuário ec2-user ao grupo docker
              usermod -aG docker ec2-user
              
              # 3. Cria um script de inicialização que roda em todo reboot
              cat << 'SCRIPT' > /var/lib/cloud/scripts/per-boot/01-update-mlflow.sh
              #!/bin/bash
              systemctl start docker
              
              echo "Limpando containers antigos..."
              docker stop mlflow-server || true
              docker rm mlflow-server || true
              docker system prune -a -f
              
              echo "Atualizando imagem do MLflow..."
              docker pull ${var.docker_image_mlflow}
              
              echo "Iniciando contêiner do MLflow apontando para S3..."
              docker run -d -p 5000:5000 \
                --name mlflow-server \
                --restart always \
                -e AWS_DEFAULT_REGION=${var.aws_region} \
                ${var.docker_image_mlflow} \
                mlflow server \
                --backend-store-uri sqlite:///mlflow.db \
                --default-artifact-root s3://fiappostech9mletgrupo17-fase02-mlflow-artifacts/ \
                --host 0.0.0.0 \
                --port 5000
              SCRIPT
              
              chmod +x /var/lib/cloud/scripts/per-boot/01-update-mlflow.sh
              
              # Executa o script para o primeiro boot
              /var/lib/cloud/scripts/per-boot/01-update-mlflow.sh
              EOF

  tags = {
    Name = "Recommender-MLflow-Server"
  }
}

# 5. AWS ACM Certificate: Solicitando certificado SSL gratuito na Virgínia
resource "aws_acm_certificate" "mlflow_cert" {
  provider          = aws.us_east_1
  domain_name       = var.mlflow_domain
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

# Recurso para aguardar a validação do certificado ACM via DNS
resource "aws_acm_certificate_validation" "mlflow_cert_validation" {
  provider        = aws.us_east_1
  certificate_arn = aws_acm_certificate.mlflow_cert.arn
}

# 6. AWS CloudFront: Distribuição HTTPS e Geo-Blocking
resource "aws_cloudfront_distribution" "mlflow_cdn" {
  enabled = true
  aliases = [var.mlflow_domain]

  origin {
    domain_name = aws_instance.mlflow_server.public_dns
    origin_id   = "EC2Origin"

    custom_origin_config {
      http_port              = 5000
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = "EC2Origin"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods         = ["GET", "HEAD"]

    forwarded_values {
      query_string = true
      headers      = ["*"]
      cookies {
        forward = "all"
      }
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "whitelist"
      locations        = ["BR", "PT"]
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate_validation.mlflow_cert_validation.certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  web_acl_id = aws_wafv2_web_acl.mlflow_waf.arn
}

# 7. AWS WAF (Web Application Firewall) para Rate Limiting
resource "aws_wafv2_web_acl" "mlflow_waf" {
  provider    = aws.us_east_1
  name        = "recommender-mlflow-waf"
  description = "WAF para bloquear abusos e ataques contra o MLflow Server"
  scope       = "CLOUDFRONT"

  default_action {
    allow {}
  }

  rule {
    name     = "RateLimitRule"
    priority = 1

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 100
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = false
      metric_name                = "RateLimitRuleMetric"
      sampled_requests_enabled   = false
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = false
    metric_name                = "RecommenderMlflowWafMetric"
    sampled_requests_enabled   = false
  }
}
