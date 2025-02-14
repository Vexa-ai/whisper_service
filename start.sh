#!/bin/bash

# Start Ray
echo "Starting Ray..."
ray start --head --node-ip-address=0.0.0.0 --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 

# Deploy the service
echo "Deploying the service..."
serve deploy app:transcriber_app

# Check if deployment was successful with retries
echo "Checking deployment status..."
max_retries=10
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if serve status | grep -q "RUNNING"; then
        echo "Service successfully deployed and running!"
        break
    fi
    echo "Waiting for service to start... ($(($retry_count + 1))/$max_retries)"
    sleep 1
    retry_count=$((retry_count + 1))
done

if [ $retry_count -eq $max_retries ]; then
    echo "Service failed to start after $max_retries retries!"
    exit 1
fi

# Keep the script running while monitoring the service
while true; do
    if ! serve status | grep -q "RUNNING"; then
        echo "Service is no longer running!"
        exit 1
    fi
    sleep 1
done 