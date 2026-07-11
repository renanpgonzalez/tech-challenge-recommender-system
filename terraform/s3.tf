resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket        = "fiappostech9mletgrupo17-fase02-mlflow-artifacts"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "mlflow_artifacts_block" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts_versioning" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}
