# Frozen GPT configuration

This package stores `config.json`, `prompt.txt`, and `empirical_prompt.txt`,
with checksums in `manifest.json`. The configured remote snapshot is
`gpt-5.4-mini-2026-03-17` through the API backend.

The remote GPT weights are unavailable locally; this package cannot perform
offline inference. It contains no API keys or credentials. The existing pipeline
accepts `--config models/gpt/config.json`.

Prompts are exact copies of the source templates identified in the manifest;
the application still uses its prompt implementation. The demonstration bank
and retrieval code remain necessary when training examples are configured.
Prompt, demonstration bank, or parameter changes require a new execution identity.
