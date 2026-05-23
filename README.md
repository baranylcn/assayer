# Assayer

[![CI](https://github.com/PracticalMind/assayer/actions/workflows/ci.yml/badge.svg)](https://github.com/PracticalMind/assayer/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/assayer)](https://pypi.org/project/assayer/)
[![Python](https://img.shields.io/pypi/pyversions/assayer)](https://pypi.org/project/assayer/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Send a prompt to multiple language models in parallel and compare their outputs in the terminal. Useful for evaluating which model handles a given task better, measuring semantic similarity between responses, or running an LLM-as-judge evaluation - without leaving the shell.

![demo](demo.gif)

## Installation

```bash
pip install assayer
```

Similarity scoring requires the optional `score` extra:

```bash
pip install "assayer[score]"
```

Python 3.11 or newer is required.

## Supported Providers

- **OpenAI**: All GPT models.
- **Anthropic**: Claude models (Opus 4.7, Sonnet 4.6, Haiku 4.5, and earlier).
- **Google Gemini**: Gemini 2.x and 3.x models.
- **Ollama**: Local models running on your machine.

## Configuration

Assayer looks for API keys in environment variables or a configuration file at `~/.assayer/config.json`.

### Environment Variables

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

### Configuration File

```json
{
  "OPENAI_API_KEY": "sk-...",
  "ANTHROPIC_API_KEY": "sk-ant-...",
  "GEMINI_API_KEY": "..."
}
```

Use `assayer models check` to verify your configuration.

## Quickstart

```bash
assayer run "Explain recursion in one sentence." --models gpt-4o,claude-haiku-4-5-20251001
```

## Commands

### `run`

```bash
assayer run "prompt" --models gpt-4o,claude-sonnet-4-5
assayer run --prompt-file prompt.txt --models gpt-4o,ollama/llama3.2
assayer run "prompt" --models gpt-4o,claude-sonnet-4-5 --score
assayer run "prompt" --models gpt-4o,claude-sonnet-4-5 --judge gpt-4o --judge-criteria "clarity,brevity"
assayer run "prompt" --models gpt-4o,claude-sonnet-4-5 --output results.json
assayer run "prompt" --models gpt-4o,claude-sonnet-4-5 --output results.csv
assayer run "prompt with {var}" --models gpt-4o --var key=value
```

| Flag | Description |
|---|---|
| `--models` | Comma-separated model identifiers (required) |
| `--prompt-file` | Path to a `.txt` file instead of an inline prompt |
| `--var` | `KEY=VALUE` template variable, repeatable |
| `--system` | System prompt applied to all models |
| `--temperature` | Sampling temperature |
| `--max-tokens` | Maximum output tokens |
| `--score` | Show pairwise similarity matrix |
| `--judge` | Model to use as judge |
| `--judge-criteria` | Comma-separated criteria for the judge |
| `--output` | Save results to `.json` or `.csv` |
| `--timeout` | Per-model timeout in seconds (default: 30) |

### `models`

```bash
assayer models list               # list all supported model identifiers
assayer models check              # check which API keys are configured
assayer models check ollama       # check if Ollama is running and list local models
```

### `config`

```bash
assayer config set OPENAI_API_KEY sk-...
assayer config show
```

Keys are saved to `~/.assayer/config.json`. Environment variables take precedence.

## Providers

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
```

Supported models: `gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4`, `gpt-5.4-pro`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.2`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`, `o3`, `o3-mini`, `o4-mini`

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Supported models: `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`, `claude-opus-4-6`, `claude-sonnet-4-5`, `claude-opus-4-5`

### Google Gemini

```bash
export GEMINI_API_KEY=...
```

Supported models: `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`, `gemini-2.0-flash-lite`

### Ollama (local)

No API key needed. Start Ollama and use the `ollama/` prefix:

```bash
ollama serve
assayer run "prompt" --models ollama/llama4-scout,ollama/llama3.2,ollama/qwen3
```

## Scoring

`--score` embeds all outputs using `all-MiniLM-L6-v2` (runs locally, no API call) and displays a pairwise cosine similarity matrix. Values range from 0 (unrelated) to 1 (identical meaning).

## LLM-as-judge

`--judge <model>` sends all outputs to the specified model and asks it to pick a winner. Use `--judge-criteria` to focus the evaluation:

```bash
assayer run "Write a sorting algorithm." \
  --models gpt-4o,claude-sonnet-4-5 \
  --judge gpt-4o \
  --judge-criteria "correctness,readability"
```

If the judge call fails, a warning is printed to stderr and the run continues normally.

## Export

`--output results.json` saves full results as JSON. `--output results.csv` saves as CSV. The file format is determined by the extension.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and the PR process.

## License

MIT - see [LICENSE](LICENSE) for details.
