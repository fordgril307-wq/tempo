#!/usr/bin/env bash
# One-command deploy from AWS CloudShell (no local tools needed).
#   1) Upload tempo.zip to CloudShell (Actions > Upload file)
#   2) unzip -o tempo.zip -d tempo && cd tempo && bash deploy/deploy.sh [alert-email]
set -euo pipefail

REGION="${AWS_REGION:-ap-northeast-2}"
STACK="tempo"
EMAIL="${1:-}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
CODE_BUCKET="tempo-code-${ACCOUNT}-${REGION}"

echo "==> Region ${REGION}, account ${ACCOUNT}"
if ! aws s3api head-bucket --bucket "$CODE_BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$CODE_BUCKET" --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  aws s3api put-public-access-block --bucket "$CODE_BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi

echo "==> Packaging application"
TMP="$(mktemp -d)"
zip -qr "$TMP/tempo.zip" Dockerfile requirements.txt tempo app samples eval/*.json -x '*/__pycache__/*'
aws s3 cp "$TMP/tempo.zip" "s3://${CODE_BUCKET}/code/tempo.zip" --region "$REGION"

echo "==> Creating/updating CloudFormation stack (about 5 minutes)"
aws cloudformation deploy --region "$REGION" --stack-name "$STACK" \
  --template-file deploy/cloudformation.yaml --capabilities CAPABILITY_IAM \
  --parameter-overrides CodeBucket="$CODE_BUCKET" AlertEmail="$EMAIL"

URL="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='Endpoint'].OutputValue" --output text)"
echo "==> Waiting for the app to build and start"
for i in $(seq 1 60); do
  if curl -fs "${URL}healthz" >/dev/null 2>&1; then
    echo; echo "Tempo is live: ${URL}"; curl -s "${URL}healthz"; echo; exit 0
  fi
  printf '.'; sleep 10
done
echo; echo "Stack is up but the app is not answering yet. Retry: curl ${URL}healthz"
