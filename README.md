# Amplifier Gemini Provider Module

Google Gemini model integration for Amplifier via Google AI API.

## Prerequisites

- **Python 3.11+**
- **[UV](https://github.com/astral-sh/uv)** - Fast Python package manager

### Installing UV

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Purpose

Provides access to Google's Gemini 2.5 models (Flash, Pro, Flash-Lite) as an LLM provider for Amplifier with 1M token context window and extended thinking capabilities.

## Contract

**Module Type:** Provider
**Mount Point:** `providers`
**Entry Point:** `amplifier_module_provider_gemini:mount`

## Supported Models

**Current support**: Text generation, tool calling, and thinking. Multimodal capabilities (images, video, audio) are not yet implemented.

### Gemini 3.0 (Latest - Preview)

- `gemini-3-pro-preview` - Best model for advanced reasoning and text generation (1M context, 65.5K output)

### Gemini 2.5 (Stable - Recommended)

- `gemini-2.5-flash` - Best price-performance for large-scale processing (1M context, default)
- `gemini-2.5-pro` - State-of-the-art thinking model for complex reasoning (1M context)
- `gemini-2.5-flash-lite` - Fastest model optimized for cost-efficiency (1M context)

### Gemini 2.0 (Stable)

- `gemini-2.0-flash` - Well-rounded capabilities with focus on price-performance (1M context)
- `gemini-2.0-flash-lite` - Optimized for cost efficiency and low latency (1M context)

**Note**: Image/video/audio models not listed as the provider doesn't support multimodal capabilities yet.

## Configuration

```toml
[[providers]]
module = "provider-gemini"
name = "gemini"
config = {
    default_model = "gemini-2.5-flash",
    max_tokens = 8192,
    temperature = 0.7,
    debug = false,
    raw_debug = false
}
```

### Debug Configuration

**Standard Debug** (`debug: true`):
- Emits `llm:request:debug` and `llm:response:debug` events
- Contains request/response summaries with truncated values (default 180 chars)
- Moderate log volume, suitable for development

**Raw Debug** (`debug: true, raw_debug: true`):
- Emits `llm:request:raw` and `llm:response:raw` events
- Contains complete, unmodified request params and response objects
- Extreme log volume, use only for deep provider integration debugging
- Captures the exact data sent to/from Gemini API before any processing

**Example**:
```yaml
providers:
  - module: provider-gemini
    config:
      debug: true           # Enable debug events
      raw_debug: true       # Enable raw API I/O capture
      debug_truncate_length: 180  # Control truncation length
      default_model: gemini-2.5-flash
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | string | env: `GOOGLE_API_KEY` | Google AI API key |
| `default_model` | string | `gemini-2.5-flash` | Default model to use |
| `max_tokens` | int | 8192 | Maximum output tokens |
| `temperature` | float | 0.7 | Sampling temperature (0.0-1.0) |
| `timeout` | float | 300.0 | API timeout in seconds |
| `priority` | int | 100 | Provider selection priority |
| `debug` | bool | false | Enable debug-level logging with truncated values |
| `raw_debug` | bool | false | Enable ultra-verbose raw API I/O logging (requires debug=true) |
| `debug_truncate_length` | int | 180 | Maximum string length in debug logs |

## Environment Variables

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

## Getting Started

### For End Users

If you have Amplifier installed via `uv tool install amplifier`, add the Gemini provider to your profiles:

1. **Set your API key**:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```
   Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

2. **Create a profile** in `~/.amplifier/profiles/gemini.md`:
   ```yaml
   ---
   profile:
     name: gemini
     version: 1.0.0
     description: Gemini provider profile
     extends: developer-expertise:dev

   providers:
     - module: provider-gemini
       source: git+https://github.com/microsoft/amplifier-module-provider-gemini@main
       config:
         default_model: gemini-2.5-flash
         temperature: 0.7
   ---
   ```

3. **Use it**:
   ```bash
   amplifier run --profile gemini "Hello from Gemini!"
   ```

### In Profile Configuration

To add Gemini to any existing profile:

```yaml
providers:
  - module: provider-gemini
    source: git+https://github.com/microsoft/amplifier-module-provider-gemini@main
    config:
      default_model: gemini-2.5-flash
      temperature: 0.7
```

### Example Profiles

**Balanced** (1M context, cost-effective):
```yaml
providers:
  - module: provider-gemini
    source: git+https://github.com/microsoft/amplifier-module-provider-gemini@main
    config:
      default_model: gemini-2.5-flash
      max_tokens: 8192
```

**Thinking** (complex reasoning):
```yaml
session:
  orchestrator:
    module: loop-streaming
    source: git+https://github.com/microsoft/amplifier-module-loop-streaming@main
    config:
      extended_thinking: true  # Show thinking content

providers:
  - module: provider-gemini
    source: git+https://github.com/microsoft/amplifier-module-provider-gemini@main
    config:
      default_model: gemini-2.5-pro
      temperature: 1.0
      max_tokens: 16384
```

**Fast** (simple queries, low cost):
```yaml
providers:
  - module: provider-gemini
    source: git+https://github.com/microsoft/amplifier-module-provider-gemini@main
    config:
      default_model: gemini-2.5-flash-lite
      temperature: 0.5
```

## Features

### Core Capabilities

- **Text Generation** - Single and multi-turn conversations
- **Tool/Function Calling** - OpenAPI schema format
- **Extended Thinking** - Reasoning with adjustable token budget
- **Streaming Support** - Incremental response generation
- **1M Token Context** - Process extremely large inputs (Flash models)
- **Message Validation** - Defense-in-depth error checking

### Thinking/Reasoning

**Gemini 2.5 models (Pro and Flash) think by default** using dynamic token budgets. The provider automatically captures thinking content from the Gemini API.

**To display thinking output**, configure your orchestrator (not the provider):

```yaml
session:
  orchestrator:
    module: loop-streaming     # Required for thinking display
    source: git+https://github.com/microsoft/amplifier-module-loop-streaming@main
    config:
      extended_thinking: true  # Show thinking content to user
```

**Model thinking behavior**:
- **gemini-2.5-pro**: Thinks by default (best for complex reasoning)
- **gemini-2.5-flash**: Thinks by default (good for most tasks)
- **gemini-2.5-flash-lite**: Does NOT think by default

**Note**: The provider captures thinking from the API automatically. The orchestrator's `extended_thinking: true` config controls whether it's displayed. Without this config, thinking still happens but isn't shown to the user.

### Tool Calling

Functions are declared using OpenAPI schema format:

```python
tools = [{
    "name": "get_weather",
    "description": "Get weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    }
}]
```

The provider handles tool call marshaling and response integration automatically.

## Known Limitations

### Synthetic Tool Call IDs

The Gemini API does not provide tool call IDs (unlike Anthropic and OpenAI). The provider generates synthetic IDs using the format `gemini_call_{uuid}` to maintain compatibility with Amplifier's tool protocol.

**Impact**: Tool call IDs are unique and functional but not provided by the API itself. This is transparent to users but documented for debugging purposes.

### Text-Only in Current Version

The provider implements text generation, tool calling, and thinking support. Multimodal capabilities (images, video, audio) are not yet supported.

## Dependencies

- `google-genai>=1.40.0` - Official Google AI Python SDK

## Development

### Local Testing with UV-Installed Amplifier Tool

When testing a local provider with a uv-installed amplifier tool (not a local build), you need to:

1. **Configure module resolution** to find your local code
2. **Install the provider's unique dependencies** into the tool's isolated Python environment

**Why this is needed**: The `amplifier` tool installed via `uv tool install` runs in an isolated Python environment at `~/.local/share/uv/tools/amplifier/`. This environment has `amplifier-core` and other core libraries, but NOT your provider's specific dependencies (like `google-genai`).

#### Step 1: Create Settings File

Create `.amplifier/settings.yaml` in the provider repository:

```yaml
# .amplifier/settings.yaml
sources:
  provider-gemini: file://.
```

This tells the module resolver to use your local code instead of fetching from git/packages.

#### Step 2: Install Provider Dependencies in Tool Environment

```bash
# Install google-genai into the amplifier tool's Python environment
uv pip install --python ~/.local/share/uv/tools/amplifier/bin/python google-genai
```

**Note**: This is a one-time setup. The dependency stays installed until you reinstall the amplifier tool.

#### Step 3: Test with Profile

```bash
# Use the included test profile
amplifier run --profile gemini-test "Hello, test the Gemini provider"

# Or create your own test profile in your project
```

**Alternative**: Use environment variables (but note they only work in the same shell session):

```bash
export AMPLIFIER_MODULE_PROVIDER_GEMINI=$(pwd)
amplifier run --profile gemini-test "test"
```

### Unit Testing (No Amplifier Tool Required)

```bash
cd amplifier-module-provider-gemini

# Install dependencies
uv sync --dev

# Run tests
uv run pytest

# Test with coverage
uv run pytest --cov
```

### Cleaning Up After Local Development

When you're done testing locally:

```bash
# Remove the settings file (if you want to use published version)
rm .amplifier/settings.yaml

# Optional: Uninstall google-genai from tool environment
# (Only if you're not testing other Google-based providers)
uv pip uninstall --python ~/.local/share/uv/tools/amplifier/bin/python google-genai
```

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
