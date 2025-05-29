#!/bin/bash
# Load environment variables from .env file
set -a
source .env
set +a
# Check if required environment variables are set
if [ -z "$PHONE_NUMBER_ID" ] || [ -z "$ACCESS_TOKEN" ]; then
  echo "Error: PHONE_NUMBER_ID and ACCESS_TOKEN must be set in the .env file."
  exit 1
fi

curl -X POST \
  "https://graph.facebook.com/v19.0/${PHONE_NUMBER_ID}/messages" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "messaging_product":"whatsapp",
        "to":"<your_mobile_number>",
        "type":"text",
        "text":{"body":"Hello from EarlyWarnBot!"}
      }'