# Amplifier Gemini Provider Module

Google Gemini model integration for Amplifier via Google AI API.

The [draft image-operation contract](contracts/image-generation.v1.md) describes
optional native image execution and its qualification requirements. It is not a
claim that the proposed Interactions adapter or capability methods are shipped.

## Optional image-operation surface — source candidate

The current source contains an experimental native image implementation in
[`image_operation.py`](amplifier_module_provider_gemini/image_operation.py).
It is separate from ordinary chat and from the explicit `image.backends`
registration described below. The draft contract remains unfrozen. Offline SDK
transport tests cover preparation, single-send execution and bounded evidence
recovery. This is not a shipped host deployment, a general multimodal-chat claim
or live account entitlement.

The ordinary provider exposes three optional structural methods without a
required Core Provider change:

```python
get_image_generation_capabilities() -> dict
async prepare_image_generation(request: dict) -> dict
async generate_image(request: dict, *, plan: dict) -> dict
```

Callers must require the complete surface and negotiate
`protocol: amplifier.image-operation.v1`. Capability inspection is local and
does not construct a client, list models or sign in. Preparation validates
supported models, transport and controls without a catalogue or generation
request; its single-use plan binds the exact request and provider connection.
The caller owns permission, durable effect identity, files and reconciliation.

The native image model is supplied by
`selection.config.image_generation: {model, transport}`. Only those two keys are
accepted in that nested operation override; credentials, endpoints, `enabled`,
timeout and nested effort are not operation payload controls.
`selection.model` remains separate conversational-role metadata; no unused
controller is called. An absent provider-level image setting does not disable
the optional surface, but an explicit `image_generation.enabled: false` does.
Canonical Developer API connection provenance is required; Vertex, Enterprise,
custom or unqualified injected connections do not inherit support.

`interactions` is the default transport. Its private SDK seam is qualified
against `google-genai` **2.30.1** and uses an isolated options view with
`HttpRetryOptions(attempts=0)`: that SDK's Interactions builder maps `attempts=1`
to two physical sends. Requests use `store: false`, non-streaming output and no
automatic continuation. Older SDKs are not silently switched to another
transport; newer versions still need the same wire-level qualification.

Explicit `transport: generate_content` is the compatibility path. It uses
`attempts=1`, one candidate, text/image modalities and disabled automatic function
calling. A request-local client view is forced to borrow the provider's HTTPX
transport, avoiding aiohttp's inner disconnect retry. Neither transport changes
the shared chat client's retries or owns cleanup of the borrowed connection.
Image timeout is the provider's image timeout when configured, otherwise the
provider timeout; null leaves no elapsed deadline.

For the optional `generate_content` path, absent, null, empty, unspecified or
unrecognized [candidate stopping evidence](https://ai.google.dev/api/generate-content#v1beta.Candidate)
means `unknown`; recognized unsuccessful reasons mean `failed`, while existing
policy-block reasons remain `blocked`. This is adapter policy, not a Google
mandate, a pollable background operation or authorization to retry. A received
response is evidence, not success; bounded image/text outputs and available
provenance remain recoverable.

The current operation accepts one final image, `auto` size/quality/background,
and one to four edit inputs, except the Flash Lite image model accepts only one.
Unsupported controls fail before dispatch. Final image/text outputs retain
order; thought output is excluded, bounded JPEG/WebP images are normalized to
PNG, and requested models, observed versions, IDs and native/normalized usage
remain distinct. Missing evidence stays null and `continuation` is null.
Malformed final parts mark the operation unsuccessful while bounded independent
image parts remain recoverable evidence, even when the malformed part came first.
Failures, cancellation and uncertain sends never authorize a second transport,
account or paid replay. This does not guarantee exactly-once service execution
or establish that a catalogue-listed model is entitled on the account.

## Quick Start

### Option 1: Install as Module (Recommended)

The simplest way to use Gemini with Amplifier. Once installed, Amplifier will automatically discover it.

1. **Set your API key**:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```
   Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

2. **Add the module**:
   ```bash
   amplifier module add provider-gemini --source git+https://github.com/microsoft/amplifier-module-provider-gemini@main --global
   ```

3. **Use Gemini as your provider**:
   ```bash
   amplifier provider use gemini --global
   ```

4. **Start using it**:
   ```bash
   amplifier run "Hello from Gemini!"
   ```

That's it! The module is now available for all your projects. Use `--project` instead of `--global` to install for just the current project.

### Option 2: Via Bundle

For more control over configuration or to compose with other capabilities, use a bundle:

1. **Set your API key**:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```
   Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

2. **Create a bundle** in your project or home directory (e.g., `gemini-bundle/bundle.md`):
   ```yaml
   ---
   bundle:
     name: gemini-dev
     version: 1.0.0
     description: Gemini provider with full 1M context

   includes:
     - bundle: foundation

   session:
     context:
       config:
         max_tokens: 1048576  # Full 1M input context

   providers:
     - module: provider-gemini
       source: git+https://github.com/microsoft/amplifier-module-provider-gemini@main
       config:
         default_model: gemini-3.8-flash
         max_output_tokens: 65536  # Full 65K output capacity
         temperature: 0.7
         priority: 50  # Lower number = higher priority (beats default 100)
   ---

   # Gemini Development Bundle

   This bundle configures Gemini with full context windows and includes foundation capabilities.

   ## Available Models

   - **Gemini 3.8 Flash** - `gemini-3.8-flash` - Default Flash model; explicit pins remain unchanged
   - **Gemini 3.5 Flash / Flash-Lite** - `gemini-3.5-flash` / `gemini-3.5-flash-lite` - Legacy Flash generation
   - **Gemini 2.5 Flash / Flash-Lite / Pro** - `gemini-2.5-flash` / `gemini-2.5-flash-lite` / `gemini-2.5-pro` - Two generations back; still served
   ```

3. **Use it**:
   ```bash
   amplifier run --bundle ./gemini-bundle "Hello from Gemini!"
   ```

## Prerequisites

Offline provider contracts run without real credentials: scoped fixtures mount
with a nonfunctional key and mock only the SDK async model pager. Run
`uv run pytest -q -m "not live"` for all offline checks.

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

Provides access to Google's Gemini models as an LLM provider for Amplifier with 1M token context windows and extended thinking capabilities.

## Contract

**Module Type:** Provider
**Mount Point:** `providers`
**Entry Point:** `amplifier_module_provider_gemini:mount`

## Supported Models

**Current support**: Text generation, tool calling, and thinking. Multimodal capabilities (images, video, audio) are not yet implemented.

Model availability and naming change frequently -- this list reflects what was verified live against `list_models()` on 2026-08-29. Always prefer `amplifier provider models gemini` (or your account's actual `list_models()` result) over this table for what your key can currently use.

### Gemini 3.x (Current generation)

- `gemini-3.8-flash` - **Default model for this provider.** Selected from the October 2026 catalog refresh; explicit pins remain unchanged. Supported thinking levels follow the provider's model-specific capability rules.
- `gemini-3.7-flash` - Previous default. Uses `thinking_level` (low/medium/high -- no `minimal`, verified in August).
- `gemini-3.5-flash` / `gemini-3.5-flash-lite` - Documented by Google as **legacy** relative to 3.7. Uses `thinking_level` (full minimal/low/medium/high range).
- Other `gemini-3.*` preview/dated ids (e.g. `gemini-3.1-flash-lite-preview`, `gemini-3-pro-image-preview`) come and go -- this provider assumes any `gemini-3.*` id supports `thinking_level` with the full range unless proven otherwise by a live 400.

### Gemini 2.5 (Two generations back; still served)

- `gemini-2.5-flash` - Best price-performance for large-scale processing (1M context, 65K max output)
- `gemini-2.5-pro` - State-of-the-art thinking model for complex reasoning (1M context, 65K max output)
- `gemini-2.5-flash-lite` - Fastest model optimized for cost-efficiency (1M context, 65K max output)

**Verified live (2026-08-29): these three REJECT `thinking_level` outright** ("Thinking level is not supported for this model") -- this provider automatically falls back to the legacy `thinking_budget` control for them. See [Thinking/Reasoning](#thinkingreasoning) below.

### Gemini 2.0

Shut down by Google -- no longer served. Do not configure `gemini-2.0-flash` / `gemini-2.0-flash-lite` as your model.

**Note**: Image/video/audio models not listed as the provider doesn't support multimodal capabilities yet.

## Configuration

```toml
[[providers]]
module = "provider-gemini"
name = "gemini"
config = {
    default_model = "gemini-3.8-flash",
    max_output_tokens = 65536,
    temperature = 0.7,
}
```

### Configuration Options

Every key below corresponds either to a real parameter in Google's `google.genai.types.GenerateContentConfig` (linked as "API: `field_name`") or is Amplifier-only glue with no Google equivalent (marked "Amplifier-only").

| Parameter | Type | Default | API param / origin | Description |
|-----------|------|---------|---------------------|-------------|
| `api_key` | string | env: `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Amplifier-only | Google AI API key. Env vars match the official SDK's own resolution (`GOOGLE_API_KEY` wins if both are set). |
| `default_model` | string | `gemini-3.8-flash` | Amplifier-only | Default model to use when a request doesn't override it. |
| `max_output_tokens` | int | model default (65,536 for every current model) | API: `max_output_tokens` | Maximum output tokens. **Renamed from `max_tokens`** to match Google's own parameter name -- `max_tokens` still works as a deprecated alias (one-shot warning; `max_output_tokens` always wins if both are set). |
| `max_tokens` | int | -- | *(deprecated alias)* | Old name for `max_output_tokens`. Prefer the new name in new configs. |
| `temperature` | float | 0.7 | API: `temperature` | Sampling temperature (0.0-2.0 per Google's docs; this provider does not clamp the range itself). |
| `timeout` | float or null | null | Provider and SDK | Optional API call / stream idle timeout in seconds; default waits for completion. |
| `close_timeout` | float | 5.0 | Amplifier-only | Hard bound on closing the genai client at session teardown, seconds. Both of the SDK client's httpx transports are closed (`client.aio.aclose()` for the async one this provider actually uses, and the synchronous `client.close()`, offloaded to a worker thread so it never blocks the event loop); neither has a deadline of its own and either can block forever on a half-closed (CLOSE-WAIT) connection. On timeout the client is abandoned with a WARNING and the sockets are reclaimed at process exit. |
| `priority` | int | 100 | Amplifier-only | Provider selection priority (lower = preferred). Read by the orchestrator's provider-selection logic, not by this module's own request-building code. |
| `raw` | bool | false | Amplifier-only | Enable raw API request/response capture on the `llm:request` / `llm:response` events (non-streaming path only). |
| `use_streaming` | bool | true | Amplifier-only | Use `generate_content_stream` instead of a single blocking `generate_content` call. Per-request override: `request.metadata["stream"] = False`. |
| `max_retries` | int | 5 | Amplifier-only | Max retry attempts on transient failures (5xx, timeouts, rate limits, Cloudflare/CDN challenges). |
| `min_retry_delay` | float | 1.0 | Amplifier-only | Initial retry backoff delay, in seconds. |
| `max_retry_delay` | float | 60.0 | Amplifier-only | Maximum retry backoff delay, in seconds. |
| `retry_jitter` | bool | true | Amplifier-only | Add random jitter to retry backoff delays. |
| `max_concurrent_requests` | int | 5 | Amplifier-only | Process-wide concurrency limit shared across all provider instances (parent + delegated sessions). `0` disables the limit. |
| `extra_request_params` | dict | `{}` | *(passthrough)* | Settings-only escape hatch -- see [extra_request_params](#extra_request_params-advanced) below. Never an interactive config field; owner-beware. |

Boolean and numeric values above tolerate string input (`"true"`/`"false"`, `"600"`, etc. -- as written by the app-cli wizard or hand-edited YAML) and warn-and-default rather than crash on anything unparseable. Unrecognized config keys log a warning at mount time (with a "did you mean" suggestion for likely typos); they never silently do nothing without a signal.

> **Removed in this revision**: `debug`, `raw_debug`, and `debug_truncate_length` were documented here in earlier README revisions but were **never actually implemented** by this provider (no code path reads them). If your config still sets them, they are now flagged with a specific "this key is inert" warning at mount time instead of being silently ignored. Use `raw: true` for this provider's actual raw-I/O capture on `llm:request`/`llm:response` events.

### extra_request_params (advanced)

`extra_request_params` merges arbitrary fields directly into the `GenerateContentConfig` this provider builds for every request -- reaching Google API parameters this provider doesn't otherwise expose as a first-class config key: `safety_settings`, `top_p`, `top_k`, `seed`, `stop_sequences`, `presence_penalty`, `frequency_penalty`, `response_mime_type`, `labels`, and anything else `google.genai.types.GenerateContentConfig` defines.

```yaml
providers:
  - module: provider-gemini
    config:
      default_model: gemini-3.8-flash
      extra_request_params:
        top_p: 0.95
        safety_settings:
          - category: HARM_CATEGORY_DANGEROUS_CONTENT
            threshold: BLOCK_ONLY_HIGH
```

**Contract:**
- Merged **LAST**, after every value this provider computes itself (temperature, `max_output_tokens`, `thinking_config`, tools). Your `extra_request_params` value always wins.
- Overriding a value this provider had already set logs a **warning** naming the field, this provider's computed value, and your override -- a silent production override never happens.
- **`thinking_config` is the nested exception:** a mapping or SDK `ThinkingConfig` merges only the fields explicitly supplied into the provider's computed typed config. An empty mapping or `ThinkingConfig()` is a no-op; explicit `false` is preserved, while explicit `null` clears that field. Both SDK spellings work (`thinking_level`/`thinkingLevel`, `thinking_budget`/`thinkingBudget`, and `include_thoughts`/`includeThoughts`). Additive and equal-value updates are quiet; conflicting replacements produce one warning listing the changed field names, without dumping their values.
- Inside that merge, `null` explicitly clears its field. A non-null `thinking_level` clears an inherited `thinking_budget`, and vice versa, because Gemini rejects requests containing both. Supplying both non-null fields together is invalid and raises `ValueError` before the request is sent. Top-level `thinking_config: null` remains a deliberate whole-field override.
- An unrecognized field name (not a real `GenerateContentConfig` field) logs a warning and is skipped -- it never crashes the provider mount.
- **Settings-only, deliberately not a `ConfigField`**: it will never appear in the interactive `amplifier init` / `amplifier provider use` wizard. Set it directly in `settings.yaml` or a bundle's config block. If your tooling round-trips provider config (e.g. re-serializing settings), this key passes through unchanged like any other dict value -- there is no special handling on this provider's side beyond the merge described above.
- **Safety filters default OFF** on Gemini 2.5/3.x models (verified against ai.google.dev) -- this provider never injects a `safety_settings` default of its own. If you want filtering, set it explicitly via `extra_request_params.safety_settings`.

## Environment Variables

The provider supports both environment variables (matching the official Google GenAI SDK):

```bash
# Either of these works (GOOGLE_API_KEY takes precedence if both are set)
export GEMINI_API_KEY="your-api-key-here"
# or
export GOOGLE_API_KEY="your-api-key-here"
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

## Usage

```python
# In amplifier configuration
[provider]
name = "gemini"
default_model = "gemini-3.8-flash"
```

## Example Bundle Configurations

For advanced configuration, add Gemini to any bundle. These examples show the YAML configuration section (the frontmatter between `---` markers in your bundle.md file):

**Basic Configuration**:
```yaml
providers:
  - module: provider-gemini
    source: git+https://github.com/microsoft/amplifier-module-provider-gemini@main
    config:
      default_model: gemini-3.8-flash
      max_output_tokens: 65536  # Use full 65K output capacity
      temperature: 0.7
      priority: 50  # IMPORTANT: Lower number = higher priority (beats default 100)
```

**Balanced** (1M context, cost-effective):
```yaml
providers:
  - module: provider-gemini
    source: git+https://github.com/microsoft/amplifier-module-provider-gemini@main
    config:
      default_model: gemini-3.8-flash
      max_output_tokens: 65536  # Full 65K output capacity
      priority: 50  # Lower number = higher priority
```

**Thinking** (complex reasoning with full 1M context):
```yaml
session:
  context:
    config:
      max_tokens: 1048576  # Full 1M input context

  orchestrator:
    module: loop-streaming
    source: git+https://github.com/microsoft/amplifier-module-loop-streaming@main
    config:
      extended_thinking: true  # Show thinking content

providers:
  - module: provider-gemini
    source: git+https://github.com/microsoft/amplifier-module-provider-gemini@main
    config:
      default_model: gemini-3.8-flash
      max_output_tokens: 65536  # Full 65K output capacity
      temperature: 1.0
      priority: 50  # Lower number = higher priority
```

**Fast** (simple queries, low cost):
```yaml
providers:
  - module: provider-gemini
    source: git+https://github.com/microsoft/amplifier-module-provider-gemini@main
    config:
      default_model: gemini-3.5-flash-lite
      max_output_tokens: 65536  # Full 65K output capacity
      temperature: 0.5
      priority: 50  # Lower number = higher priority
```

## Features

### Core Capabilities

- **Text Generation** - Single and multi-turn conversations
- **Tool/Function Calling** - OpenAPI schema format
- **Extended Thinking** - Reasoning via `thinking_level` (current models) or `thinking_budget` (legacy models)
- **Streaming Support** - Incremental response generation
- **1M Token Context** - Process extremely large inputs (Flash models)
- **Message Validation** - Defense-in-depth error checking

### Thinking/Reasoning

Google's thinking control surface changed generations: `thinking_budget` (an approximate output-token budget) is the **legacy** control; `thinking_level` (an enum: `minimal`/`low`/`medium`/`high`) is the **current** control -- and, as of this revision, the *only* control some models accept at all. Sending both on one request is rejected by the API with a 400.

This provider maps Amplifier's portable `reasoning_effort` request field to `thinking_level` automatically, per-model, clamped against a small maintained support table:

| Model | Supported `thinking_level` values | Notes |
|-------|-----------------------------------|-------|
| `gemini-3.7-flash` | `low`, `medium`, `high` | **Rejects `minimal`** (verified live: "Thinking level MINIMAL is not supported for this model"). Google's own default (when omitted) is `medium`. |
| `gemini-3.5-flash`, `gemini-3.5-flash-lite` | `minimal`, `low`, `medium`, `high` | Full range. |
| Other `gemini-3.*` ids | `minimal`, `low`, `medium`, `high` (assumed) | Not individually verified -- assumed full range until a live 400 proves narrower. |
| `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.5-flash-lite` | **none** | Rejects `thinking_level` entirely (verified live: "Thinking level is not supported for this model"). Falls back to the legacy `thinking_budget` mapping below. |
| `gemini-2.0-*` | **none** (shut down) | Not servable at all. |

`reasoning_effort` -> `thinking_level` mapping (for models that support it):

| `reasoning_effort` | Target level | If not supported by the model |
|---------------------|--------------|--------------------------------|
| `none` | *(no level sent)* | Uses `minimal` if the model supports it, otherwise the model's own default (Gemini 3.x cannot disable thinking at all -- verified live) |
| `minimal` | `minimal` | Clamped up to the nearest supported level (e.g. `low` on `gemini-3.7-flash`), logged at INFO |
| `low` | `low` | Clamped as above |
| `medium` | `medium` | Clamped as above |
| `high`, `xhigh`, `max` | `high` | Gemini has no level above `high` |

For models with **no** `thinking_level` support (the `gemini-2.5-*` family), `reasoning_effort` instead maps to the legacy numeric `thinking_budget`: `none` -> `0` (disabled), `minimal`/`low` -> `4096`, `medium`/`high`/`xhigh`/`max` -> `-1` (dynamic, model decides).

An explicit `thinking_budget` passed via `request.metadata["thinking_budget"]` or a provider `**kwargs` override always wins outright and is sent **alone** -- never combined with `thinking_level` on the same request.

**To display thinking output**, configure your orchestrator (not the provider):

```yaml
session:
  orchestrator:
    module: loop-streaming     # Required for thinking display
    source: git+https://github.com/microsoft/amplifier-module-loop-streaming@main
    config:
      extended_thinking: true  # Show thinking content to user
```

**Note**: The provider captures thinking from the API automatically (`include_thoughts` defaults to `true`). The orchestrator's `extended_thinking: true` config controls whether it's *displayed*. Without this config, thinking still happens but isn't shown to the user.

**Thought signatures**: Gemini 2.5+ models attach an opaque `thought_signature` to parts that follow a thinking burst. Because this provider is **stateless full-resend** (the entire conversation is rebuilt from stored history and resent on every turn, with no server-side session), these signatures must be captured and replayed unmodified or the API returns `FinishReason MISSING_THOUGHT_SIGNATURE`. This provider captures and echoes signatures on text, thinking, and tool-call parts alike, encoded as base64 so the value survives any JSON serialization the conversation history passes through (e.g. session persistence, event logging).

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

## Graceful Error Recovery

The provider implements automatic repair for incomplete tool call sequences:

**The Problem**: If tool results are missing from conversation history (due to context compaction bugs, parsing errors, or state corruption), the Gemini API rejects the entire request, breaking the user's session.

**The Solution**: The provider automatically detects and repairs missing tool_results by injecting synthetic results:

1. **Repair before API call** - Detects missing tool_results and injects synthetic ones
2. **Make failures visible** - Synthetic results contain `[SYSTEM ERROR: Tool result missing]` messages
3. **Maintain conversation validity** - API accepts repaired messages, session continues
4. **Enable recovery** - LLM acknowledges error and can ask user to retry
5. **Provide observability** - Emits `provider:tool_sequence_repaired` event with repair details

**Example**:
```python
# Conversation with missing tool result
messages = [
    {
        "role": "assistant",
        "content": [
            {"type": "tool_call", "id": "gemini_call_abc123", "name": "get_weather", "input": {...}}
        ]
    },
    # MISSING: {"role": "tool", "tool_call_id": "gemini_call_abc123", "content": "..."}
    {"role": "user", "content": "Thanks"}
]

# Provider repairs by injecting synthetic result:
{
    "role": "tool",
    "tool_call_id": "gemini_call_abc123",
    "name": "get_weather",
    "content": "[SYSTEM ERROR: Tool result missing from conversation history]\n\nTool: get_weather\nCall ID: gemini_call_abc123\n\nThis indicates the tool result was lost after execution.\nLikely causes: context compaction bug, message parsing error, or state corruption.\n\nThe tool may have executed successfully, but the result was lost.\nPlease acknowledge this error and offer to retry the operation."
}
```

This is a **defense-in-depth safety net**. The orchestrator should handle tool execution errors at runtime, so this repair only triggers when results go missing due to bugs in context management.

## Known Limitations

### Synthetic Tool Call IDs

The Gemini API does not provide tool call IDs (unlike Anthropic and OpenAI). The provider generates synthetic IDs using the format `gemini_call_{uuid}` to maintain compatibility with Amplifier's tool protocol.

**Impact**: Tool call IDs are unique and functional but not provided by the API itself. This is transparent to users but documented for debugging purposes.

### Text-Only in Current Version

The provider implements text generation, tool calling, and thinking support. Multimodal capabilities (images, video, audio) are not yet supported.

### Gemini 3.x Thinking Cannot Be Disabled

Verified live: `thinking_budget=0` on `gemini-3.7-flash` still produced thinking tokens, and there is no "off" `thinking_level`. Thinking is mandatory for Gemini 3.x models regardless of what this provider sends. `reasoning_effort="none"` on these models falls back to the model's own default thinking amount rather than actually disabling it -- this is a vendor limitation, not something this provider can work around.

### Native preflight counting is Gemini Developer API-only

For a configured, limit-tabled `gemini-*` Developer API model, `request_budget()` can call
`POST /v1beta/models/{model}:countTokens` with the documented nested
`generateContentRequest` body. It uses the same provider-owned request plan as
generation and includes converted system/developer messages, inline images,
thought signatures, function declarations/results, `toolConfig`,
`safetySettings`, `generationConfig`, and `cachedContent`.

The nested `generateContentRequest` carries its own REQUIRED `model` field
(`models/<selected model>`), per the REST `CountTokensRequest` contract. It is
always built from the same selected model as the `:countTokens` endpoint path,
so the URL and the counted body can never name different models.

When `request_budget()` returns no decision, it also makes a best-effort
`provider:request_budget_unavailable` observability event carrying
`provider`, `method`, an optional selected `model`, a fixed `reason` code, and an
integer `http_status` when one exists. The reason codes are
`unsupported_model`, `unsupported_route`, `request_plan_unavailable`,
`request_projection_unavailable`, `invalid_output_limit`, `http_error`, and
`invalid_response`. Exactly one event is emitted per failed budget call, and a
successful count emits none. Nothing else travels with it: no exception text,
response body, request body, URL, credential, or environment value, and a
model ID is included only when it exactly matches a documented ID in the
provider's limit table. Unknown strings, URLs, suffixed aliases and non-string
values are omitted rather than coerced; limit lookup behavior is unchanged.
A failing subscriber cannot turn a count into a product failure; with no hooks
channel at all the same reason is logged as a warning instead.

The counter uses the response's `totalTokens` as the input measurement. It does
not add `cachedContentTokenCount` a second time, does not estimate output, and
does not increase the documented raw input limit when a caller lowers
`max_output_tokens`. A missing/malformed response, HTTP failure, or request
that cannot be publicly serialized returns no budget decision rather than a
partial native count. This includes the Developer-unsupported or
transformer-specific `labels`, routing/model-selection, response-schema,
speech/image/audio, and Model Armor extras. Cancellation propagates without
starting generation. An unknown model ID is also unavailable: an exact count
cannot justify an inferred input-window admission limit. An `extra_request_params`
override for `tools` or `system_instruction`, and a noncanonical
`cached_content` value, are likewise unavailable rather than reimplementing
the SDK's private normalization.

This adapter is not Vertex AI support and does not use
`client.models.count_tokens`: that Python helper cannot send a full Gemini
Developer `generateContentRequest`. The request-projection wire regression was
originally written against `google-genai` 2.23.0. The current dependency floor and
lock are 2.30.1 for the optional image-operation SDK boundary; the existing
generation/count tests remain part of the offline suite.

Counting is unavailable when the SDK would route generation away from the
canonical Gemini Developer endpoint: a nonempty `GOOGLE_GEMINI_BASE_URL`, or
effective Vertex/Enterprise selection through `GOOGLE_GENAI_USE_VERTEXAI` /
`GOOGLE_GENAI_USE_ENTERPRISE`, disables both count advertisement and dispatch.
The SDK's 2.23 parsing is followed exactly: `true`/`1` are case-insensitive,
and a present Enterprise variable wins over Vertex even when it is explicitly
false. A client constructed by this provider retains its route decision; an
injected client has no public base-URL guarantee and therefore fails closed.
`get_info()` reads only these route-selection variables and never constructs
an SDK client, authenticates, or performs I/O.

## Dependencies

- `google-genai>=2.30.1,<3` - Official Google AI Python SDK; 2.30.1 supplies the qualified Interactions image-operation boundary. It also retains the full thinking-level enum required by ordinary chat. New SDK versions require the same transport regressions.
- `httpx>=0.28.1` - Async transport for the documented Developer
  `countTokens` REST endpoint.

## Development

### Local Testing with Installed Amplifier CLI

Test your local provider changes with the installed `amplifier` CLI:

```bash
# From the provider repository root
cd amplifier-module-provider-gemini

# Add your local provider (use --local for development, not --global)
amplifier module add provider-gemini --source file://. --local

# Set your API key
export GOOGLE_API_KEY="your-api-key-here"

# Now you can use amplifier init and see your local provider in the menu
amplifier init

# Or configure it directly
amplifier provider use gemini --local

# Test your local changes
amplifier run "Hello, testing local Gemini provider!"

# List modules to verify your local provider is registered
amplifier module list -t provider
```

**When you're done testing:**

```bash
# Remove the local provider registration
amplifier module remove provider-gemini --local
```

**Why `--local` instead of `--global`?**
- `--local` registers the provider only for the current working directory
- `--global` would affect all your projects (not ideal during development)
- `--project` works if you want to share with your team

### Unit Testing (Fastest)

For rapid iteration without the CLI:

```bash
cd amplifier-module-provider-gemini

# Install dependencies
uv sync --dev

# Run validation tests (protocol compliance)
uv run pytest

# Test with coverage
uv run pytest --cov
```

These tests validate the provider implements the required protocol without needing the full CLI.

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

### Waiting for model work

The default `timeout` is `null`: model work waits for completion, explicit cancellation, or a provider/transport error. Set a numeric `timeout` in seconds to opt into a deadline. Existing cleanup limits are unchanged.


## Native image generation and editing

The provider can supply an `image.backends` capability to a separately mounted
image tool. It remains opt-in; chat model selection does not enable image calls.

```yaml
config:
  image_generation:
    enabled: true
    id: images
    model: auto
```

`auto` selects the maintained stable image default available in this account's
model catalog. An explicit catalog model preserves a fixed choice. Catalog access
does not establish generation entitlement. `get_image_generation_info()` and
`list_image_models()` expose discovery without generating an image.

The native backend supports generation and reference-image editing, with automatic
size, quality and background. Unsupported explicit controls are rejected. It makes
one SDK attempt, returns one PNG artifact (normalizing native JPEG/WebP output),
and includes resolved model, request identity, usage and accompanying text. Thought
images are excluded. Output is limited to 16 megapixels and 8 MiB. Failures do not
trigger an automatic retry or a different paid backend. The optional `timeout`
is positive finite seconds; the default leaves the provider request unbounded.
