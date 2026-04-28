# Project Brief

## Goal

Reproduce David Noel Ng's RYS (Repeat Your Self) relayering workflow from the public write-up and released code, then package it so it can serve as the first stage of a course project on software-engineering agents.

## What is implemented

- canonical `(i, j)` and explicit layer-list relayer configs
- dense and MoE layer-duplication wrappers with shared weights
- queue-based scan setup
- Math and EQ probe workers
- balanced Math+EQ analysis and heatmap rendering
- beam search and surrogate-search utilities
- Hugging Face checkpoint export for relayered models

## What was added in this build

- a Hugging Face combined worker for local validation without ExLlama
- a one-command smoke demo using a tiny random Llama fixture
- an agent-study scaffold for baseline-vs-RYS comparisons on software-agent runs

## Validation status

- unit/integration tests: `10 passed`
- local smoke scan: completed successfully over a six-layer full queue
- local agent-study summary tooling: validated on sample run records

## Recommended next experiment

1. Pick a base instruct model that is strong enough to act inside a coding agent.
2. Run a RYS sweep or targeted scan with this repo to identify 1-3 promising blocks.
3. Export the relayered checkpoints.
4. Serve baseline and RYS variants through the same agent framework.
5. Compare task success rate, steps, execution errors, and runtime on the same issue subset.
