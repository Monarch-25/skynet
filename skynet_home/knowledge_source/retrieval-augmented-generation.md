---
title: "Retrieval-Augmented Generation (RAG)"
topic: "machine learning systems"
---

# Retrieval-Augmented Generation

## Overview

Retrieval-Augmented Generation, or RAG, grounds a language model's output in an external corpus. Instead of relying on parametric memory stored in the model weights, RAG retrieves relevant chunks from a document store at inference time and conditions the generator on those chunks.

## Why RAG

Two failure modes of pure parametric LLMs motivate RAG:
1. **Hallucination.** Without grounding, an LLM may generate fluent but false text.
2. **Staleness.** A model's parametric knowledge is frozen at training time; it cannot know about new facts.

By retrieving at inference, RAG gives the model access to fresh, source-grounded context, and lets it cite where its answer came from.

## The retrieval step

The retrieval step is usually a similarity search over dense embeddings (e.g., cosine over sentence-transformer vectors) or a sparse lexical match such as BM25. Dense retrieval handles synonyms and paraphrase well; BM25 excels on rare terms and exact phrase matches. Hybrid retrieval (BM25 + dense, fused via reciprocal rank fusion) often beats either alone.

## Chunking

Documents are split into chunks of a few hundred tokens with some overlap so that a relevant passage is not cut in half. Chunk size and overlap are hyperparameters: too small and a chunk loses context, too large and the signal-to-noise ratio drops.

## Provenance

A key practical advantage of RAG is provenance: each generated claim can be traced back to the specific chunk the model conditioned on. This makes outputs auditable and lets a user verify the source.
