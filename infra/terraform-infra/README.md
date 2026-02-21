# Terraform Infrastructure

AWS Lambda proxy that forwards requests to Modal model endpoints.

## Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/install) (`brew install terraform`)
- [AWS CLI](https://aws.amazon.com/cli/) configured with credentials (`aws login`)

## Structure

```
terraform-infra/
├── modules/
│   └── api-server/       # Shared Lambda + Function URL module
├── environments/
│   ├── dev/              # Dev environment (t3.micro, 128MB Lambda)
│   └── prod/             # Prod environment (t3.small, 256MB Lambda)
```

## Usage

### Deploy dev

```bash
cd environments/dev
terraform init
terraform plan
terraform apply
```

### Deploy prod

```bash
cd environments/prod
terraform init
terraform plan
terraform apply
```

After `terraform apply`, the Lambda URL will be printed:

```
lambda_url = "https://xxxxxxx.lambda-url.us-east-1.on.aws/"
```

### View current resources

```bash
terraform state list       # List all managed resources
terraform show             # Show full details
terraform output           # Show outputs (Lambda URL, function name)
```

### Tear down

```bash
terraform destroy
```
