# Gemini Project Guidelines

This file documents project-specific instructions, conventions, and guidelines.

## Environment & Execution
- **Environment Manager**: Use `uv` for dependency management.
- **Dependency Addition**: ALWAYS use `uv add <package>` to install new dependencies.
- **Activation**: Always activate the `uv` virtual environment or use `uv run` for executing scripts.
- **Linting**: Do NOT run the linter automatically after completing a task.

## Tokenizer Integration
- **Custom Tokenizer Path**: `indic_voices_tokenizer` (root of the folder).
- **Type**: SentencePiece (load via `LlamaTokenizer`, NOT `AutoTokenizer`).
- **Vocab Size**: ~256 (plus special tokens).
- **Language**: Manipuri (mni).
- **Model adaptation**: Requires `resize_token_embeddings` and updating `decoder_start_token_id`.
