# Gemini Provider Test Configuration

This directory contains test profiles and configuration for the Gemini provider.

## Profiles

### `gemini-test.md`

A test profile configured to use the local Gemini provider with debug logging enabled.

**Usage from this repository:**

```bash
# Basic test
amplifier run --profile gemini-test "Hello, can you test the Gemini provider?"

# Test with tool usage
amplifier run --profile gemini-test "List the files in the current directory"

# Test thinking/reasoning (edit profile to use gemini-2.5-pro first)
amplifier run --profile gemini-test "Explain the design patterns in this codebase"
```

**Configuration:**
- Provider source: `file://.` (uses local provider code)
- Debug logging: Enabled by default
- Raw debug: Disabled (edit profile to enable ultra-verbose logging)
- Model: `gemini-2.5-flash` (balanced performance)

## Environment Setup

Before testing, ensure you have:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

## Testing Checklist

- [ ] Basic text generation works
- [ ] Tool calling (bash, filesystem) works
- [ ] Debug events are emitted (check logs)
- [ ] Token counting is accurate
- [ ] Thinking/reasoning works (with gemini-2.5-pro)
- [ ] Error handling is graceful

## Debug Logging

The test profile has `debug: true` enabled. This emits:
- `llm:request` - Request summaries
- `llm:request:debug` - Detailed requests (truncated)
- `llm:response` - Response summaries
- `llm:response:debug` - Detailed responses (truncated)

For ultra-verbose logging, edit the profile to set `raw_debug: true`.
