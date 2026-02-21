terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# IAM role for Lambda
resource "aws_iam_role" "lambda" {
  name = "openneuro-${var.environment}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })

  tags = { Name = "openneuro-${var.environment}-lambda-role" }
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Placeholder Lambda function
data "archive_file" "lambda_placeholder" {
  type        = "zip"
  output_path = "${path.module}/lambda.zip"

  source {
    content  = <<-EOF
      import json
      from urllib import request, error

      MODAL_BASE_URL = "${var.modal_base_url}"

      def handler(event, context):
          path = event.get("rawPath", "/")
          method = event["requestContext"]["http"]["method"]
          headers = event.get("headers", {})
          body = event.get("body", None)

          url = MODAL_BASE_URL + path
          query = event.get("rawQueryString", "")
          if query:
              url += "?" + query

          req_headers = {
              k: v for k, v in headers.items()
              if k.lower() not in ("host", "content-length")
          }
          req_headers["Host"] = MODAL_BASE_URL.split("//")[1]

          if body and event.get("isBase64Encoded"):
              import base64
              body = base64.b64decode(body)
          elif body:
              body = body.encode("utf-8")

          req = request.Request(url, data=body, headers=req_headers, method=method)

          try:
              with request.urlopen(req, timeout=25) as resp:
                  resp_body = resp.read().decode("utf-8")
                  resp_headers = dict(resp.headers)
                  return {
                      "statusCode": resp.status,
                      "headers": resp_headers,
                      "body": resp_body,
                  }
          except error.HTTPError as e:
              resp_body = e.read().decode("utf-8")
              return {
                  "statusCode": e.code,
                  "headers": dict(e.headers),
                  "body": resp_body,
              }
    EOF
    filename = "handler.py"
  }
}

resource "aws_lambda_function" "api" {
  function_name    = "openneuro-${var.environment}-api"
  role             = aws_iam_role.lambda.arn
  handler          = "handler.handler"
  runtime          = "python3.12"
  timeout          = 30
  memory_size      = var.lambda_memory
  filename         = data.archive_file.lambda_placeholder.output_path
  source_code_hash = data.archive_file.lambda_placeholder.output_base64sha256

  tags = { Name = "openneuro-${var.environment}-api" }
}

# Lambda Function URL (direct public URL, no API Gateway needed)
resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "NONE"
}
