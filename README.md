---
title: Gait Pass ML Service
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Gait-Pass ML Service

Face recognition service using InsightFace.

## API Endpoints

- `POST /extract-embedding` - Extract face embedding from image
- `POST /compare-embeddings` - Compare two embeddings
- `POST /batch-recognize` - Recognize face from known database
- `GET /health` - Health check
