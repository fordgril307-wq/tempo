"""AWS integration: S3 for vision results / replays, SNS for safety alerts.

Without the environment variables everything runs locally (no-ops), so the
same code runs on a laptop and on AWS Graviton.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("tempo.cloud")
S3_BUCKET = os.getenv("TEMPO_S3_BUCKET")
SNS_TOPIC = os.getenv("TEMPO_SNS_TOPIC_ARN")
REGION = os.getenv("AWS_REGION", "ap-northeast-2")
_c: dict = {}


def _client(name: str):
    if name not in _c:
        import boto3
        _c[name] = boto3.client(name, region_name=REGION)
    return _c[name]


def mode() -> dict:
    return {"s3": bool(S3_BUCKET), "sns": bool(SNS_TOPIC), "region": REGION}


def put_json(key: str, obj) -> bool:
    if not S3_BUCKET:
        return False
    try:
        _client("s3").put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(obj).encode(),
                                 ContentType="application/json")
        return True
    except Exception as e:      # storage must never break perception
        log.warning("s3 put failed %s: %s", key, e)
        return False


def alert(subject: str, payload: dict) -> bool:
    log.info(json.dumps({"event": "alert", "subject": subject, **payload}))
    if not SNS_TOPIC:
        return False
    try:
        _client("sns").publish(TopicArn=SNS_TOPIC, Subject=subject[:100], Message=json.dumps(payload, indent=1))
        return True
    except Exception as e:
        log.warning("sns publish failed: %s", e)
        return False
