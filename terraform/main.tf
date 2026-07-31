terraform {
  backend "s3" {
    bucket = "fiappostech9mletgrupo17-fase02-tfstate-967982352747"
    key    = "terraform.tfstate"
    region = "sa-east-1"
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Provider específico para o ACM (A AWS exige que certificados do CloudFront fiquem na Virgínia)
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}
