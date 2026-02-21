output "lambda_url" {
  description = "Public URL of the Lambda function"
  value       = aws_lambda_function_url.api.function_url
}

output "lambda_function_name" {
  description = "Name of the Lambda function"
  value       = aws_lambda_function.api.function_name
}
