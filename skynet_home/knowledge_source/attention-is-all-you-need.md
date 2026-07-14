---
title: "Attention Is All You Need"
authors: ["Vaswani et al."]
year: 2017
---

# Attention Is All You Need

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

## 1. Introduction

Recurrent neural networks, long short-term memory, and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling. Such recurrent models typically factor computation along the symbol positions of the input and output sequences. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths.

The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality.

## 2. Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural networks as basic building blocks. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between them.

## 3.2.1: Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention". The input consists of queries and keys of dimension d_k, and values of dimension d_v. We compute the dot products of the query with all keys, divide each by the square root of d_k, and apply a softmax function to obtain the weights on the values.

The two most commonly used attention functions are additive attention, and dot-product (multiplicative) attention. Dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

## 3.2.2: Multi-Head Attention

Instead of performing a single attention function with d_model-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times to different learned projections. The multi-head attention mechanism allows the model to jointly attend to information from different representation subspaces at different positions.

## 3.4: Embeddings and Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model. The embedding weights are multiplied by the square root of d_model.

## 5.4: Layer Normalization

We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized. For the base model, we use a rate of P_drop = 0.1. The Transformer follows this architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.
