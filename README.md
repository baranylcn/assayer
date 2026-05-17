# skate

Run a prompt across multiple LLMs and compare outputs side by side in the terminal.

## Installation

```bash
pip install skate
```

Similarity scoring requires the optional `score` extra:

```bash
pip install "skate[score]"
```

Python 3.11 or newer is required.

> **Contributing?** See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, code style, and PR guidelines.

## Supported Providers

Skate supports multiple LLM providers through a unified interface:

- **OpenAI**: All GPT models.
- **Anthropic**: Claude 4.5 models (Opus, Sonnet, Haiku).
- **Google Gemini**: 1.5 Pro and Flash models.
- **Ollama**: Local models running on your machine.

## Configuration

Skate looks for API keys in environment variables or a configuration file located at `~/.skate/config.json`.

### Environment Variables

You can set your keys as standard environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

### Configuration File

Alternatively, create a file at `~/.skate/config.json`:

```json
{
  "OPENAI_API_KEY": "sk-...",
  "ANTHROPIC_API_KEY": "sk-ant-...",
  "GEMINI_API_KEY": "..."
}
```

Use `skate models check` to verify your configuration.

## Quickstart

```bash
skate run "Explain recursion in one sentence." --models gpt-4o,claude-haiku-4-5-20251001
```

## Commands

### run

```bash
skate run "prompt" --models gpt-4o,claude-sonnet-4-5
skate run --prompt-file prompt.txt --models gpt-4o,ollama/llama3
skate run "prompt" --models gpt-4o,claude-sonnet-4-5 --score
skate run "prompt" --models gpt-4o,claude-sonnet-4-5 --judge gpt-4o --judge-criteria "clarity,brevity"
skate run "prompt" --models gpt-4o,claude-sonnet-4-5 --output results.json
skate run "prompt" --models gpt-4o,claude-sonnet-4-5 --output results.csv
skate run "prompt with {var}" --models gpt-4o --var key=value
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

### models

```bash
skate models list               # list all supported model identifiers
skate models check              # check which API keys are configured
skate models check ollama       # check if Ollama is running and list local models
```

### config

```bash
skate config set OPENAI_API_KEY sk-...
skate config show
```

Keys are saved to `~/.skate/config.json`. Environment variables take precedence.

## Providers

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
```

Supported models: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `o1-mini`

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Supported models: `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-4-5-20251001`

### Google Gemini

```bash
export GEMINI_API_KEY=...
```

Supported models: `gemini-1.5-pro`, `gemini-1.5-flash`

### Ollama (local)

No API key needed. Start Ollama and use the `ollama/` prefix:

```bash
ollama serve
skate run "prompt" --models ollama/llama3,ollama/mistral,ollama/phi3
```

## Scoring

`--score` embeds all outputs using `all-MiniLM-L6-v2` (runs locally, no API call) and displays a pairwise cosine similarity matrix. Values range from 0 (unrelated) to 1 (identical meaning).

## LLM-as-judge

`--judge <model>` sends all outputs to the specified model and asks it to pick a winner. Use `--judge-criteria` to focus the evaluation:

```bash
skate run "Write a sorting algorithm." \
  --models gpt-4o,claude-sonnet-4-5 \
  --judge gpt-4o \
  --judge-criteria "correctness,readability"
```

If the judge call fails, a warning is printed to stderr and the run continues normally.

## Export

`--output results.json` saves full results as JSON. `--output results.csv` saves as CSV. The file format is determined by the extension.
