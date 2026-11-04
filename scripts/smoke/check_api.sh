#!/usr/bin/env bash
#
# CI smoke test: check the running API's contract (not its exact scores, which
# depend on the synthetic training data).
#
#   scripts/smoke/check_api.sh [base-url]     # default http://localhost:8000

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
cd "$(dirname "$0")/../.."

# Print the step, then fail loudly with the response if a check doesn't hold.
check() {
    local name="$1" body="$2" filter="$3"
    if jq -e "$filter" <<<"$body" >/dev/null; then
        echo "PASS  $name"
    else
        echo "FAIL  $name"
        echo "      response: $body"
        exit 1
    fi
}

ready=$(curl --fail-with-body --silent --show-error "$BASE_URL/health/ready")
check "readiness: model loaded" "$ready" '.status == "ready" and .model_loaded == true'

info=$(curl --fail-with-body --silent --show-error "$BASE_URL/v1/model-info")
check "model-info: telco-churn@champion" "$info" \
    '.model_name == "telco-churn" and .model_alias == "champion"'

prediction=$(curl --fail-with-body --silent --show-error \
    -X POST "$BASE_URL/v1/predict" \
    -H 'Content-Type: application/json' \
    --data-binary @examples/customer.json)
check "predict: score in [0, 1]" "$prediction" \
    '.churn_score >= 0 and .churn_score <= 1'
check "predict: label Yes/No" "$prediction" \
    '.churn_prediction == "Yes" or .churn_prediction == "No"'
version=$(jq -r '.model_version' <<<"$info")
check "predict: served by champion v$version" "$prediction" \
    ".model_version == \"$version\""

echo "Smoke test passed: $prediction"
