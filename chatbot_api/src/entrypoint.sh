#!/bin/bash
set -e

echo "$(date): Starting hospital RAG FastAPI service..."

# Start the FastAPI app with Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
