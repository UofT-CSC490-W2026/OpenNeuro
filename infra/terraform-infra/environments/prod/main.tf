provider "aws" {
  region = "us-east-1"
}

module "api_server" {
  source = "../../modules/api-server"

  environment    = "prod"
  aws_region     = "us-east-1"
  lambda_memory  = 256
  modal_base_url = "https://xingjianll--openneuro-vlm-inference-serve.modal.run"
}

output "lambda_url" {
  value = module.api_server.lambda_url
}

output "lambda_function_name" {
  value = module.api_server.lambda_function_name
}
