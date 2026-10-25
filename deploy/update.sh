#!/usr/bin/env bash
# Ship new code to the running Tempo instance (same URL, no downtime beyond a container restart).
#   1) Upload the new tempo.zip to CloudShell (Actions > Upload file)
#   2) rm -rf tempo && unzip -o tempo.zip -d tempo && cd tempo && bash deploy/update.sh
set -euo pipefail

REGION="${AWS_REGION:-ap-northeast-2}"
STACK="tempo"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
CODE_BUCKET="tempo-code-${ACCOUNT}-${REGION}"
out() { aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
          --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }

echo "==> Packaging application"
TMP="$(mktemp -d)"
zip -qr "$TMP/tempo.zip" Dockerfile requirements.txt tempo app samples eval/*.json -x '*/__pycache__/*'
aws s3 cp "$TMP/tempo.zip" "s3://${CODE_BUCKET}/code/tempo.zip" --region "$REGION"

INSTANCE="$(aws cloudformation describe-stack-resource --region "$REGION" --stack-name "$STACK" \
  --logical-resource-id Instance --query StackResourceDetail.PhysicalResourceId --output text)"
BUCKET="$(out BucketName)"; TOPIC="$(out TopicArn)"; URL="$(out Endpoint)"

echo "==> Rebuilding the container on ${INSTANCE} (about 3 minutes)"
SCRIPT="set -eux; cd /opt/tempo; aws s3 cp s3://${CODE_BUCKET}/code/tempo.zip code.zip --region ${REGION}; \
rm -rf app tempo samples eval; unzip -o -q code.zip; docker build -q -t tempo .; docker rm -f tempo || true; \
docker run -d --restart unless-stopped --name tempo -p 80:8000 -e AWS_REGION=${REGION} -e TEMPO_S3_BUCKET=${BUCKET} \
-e TEMPO_SNS_TOPIC_ARN=${TOPIC} --log-driver awslogs --log-opt awslogs-region=${REGION} \
--log-opt awslogs-group=/tempo/app --log-opt awslogs-create-group=false tempo"
CMD="$(aws ssm send-command --region "$REGION" --instance-ids "$INSTANCE" --document-name AWS-RunShellScript \
  --parameters "$(jq -cn --arg s "$SCRIPT" '{commands:[$s]}')" --timeout-seconds 900 --query Command.CommandId --output text)"
for i in $(seq 1 90); do
  ST="$(aws ssm get-command-invocation --region "$REGION" --command-id "$CMD" --instance-id "$INSTANCE" \
        --query Status --output text 2>/dev/null || echo Pending)"
  case "$ST" in
    Success) break ;;
    Failed|Cancelled|TimedOut)
      aws ssm get-command-invocation --region "$REGION" --command-id "$CMD" --instance-id "$INSTANCE" \
        --query StandardErrorContent --output text | tail -20; exit 1 ;;
  esac
  printf '.'; sleep 10
done
echo
for i in $(seq 1 30); do
  if curl -fs "${URL}healthz" >/dev/null 2>&1; then echo "Tempo updated: ${URL}"; curl -s "${URL}healthz"; echo; exit 0; fi
  printf '.'; sleep 5
done
echo "Container restarted but not answering yet. Retry: curl ${URL}healthz"
