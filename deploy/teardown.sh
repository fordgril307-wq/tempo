#!/usr/bin/env bash
# Remove everything after judging (stops all charges).
set -euo pipefail
REGION="${AWS_REGION:-ap-northeast-2}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="$(aws cloudformation describe-stacks --region "$REGION" --stack-name tempo \
  --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text || true)"
[ -n "$BUCKET" ] && aws s3 rm "s3://$BUCKET" --recursive --region "$REGION" || true
aws cloudformation delete-stack --region "$REGION" --stack-name tempo
aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name tempo
aws s3 rb "s3://tempo-code-${ACCOUNT}-${REGION}" --force --region "$REGION" || true
echo "Tempo removed."
