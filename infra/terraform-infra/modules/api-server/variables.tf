variable "environment" {
  description = "Environment name (dev or prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "lambda_memory" {
  description = "Lambda memory in MB"
  type        = number
  default     = 128
}

variable "modal_base_url" {
  description = "Base URL of the Modal deployment to proxy to"
  type        = string
}
