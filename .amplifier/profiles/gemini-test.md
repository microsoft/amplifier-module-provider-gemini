---
profile:
  name: gemini-test
  version: 1.0.0
  description: Test profile for Gemini provider with debug logging enabled
  extends: developer-expertise:dev

providers:
  - module: provider-gemini
    config:
      default_model: gemini-2.5-flash
      max_tokens: 8192
      temperature: 0.7
      timeout: 300.0
      debug: true           # Enable debug-level logging
      raw_debug: true       # Enable ultra-verbose raw API logging
      debug_truncate_length: 180
---

# Gemini Provider Test Profile

This profile is configured to test the local Gemini provider with debug logging enabled.

## Usage

**IMPORTANT**: To test the local provider code, set the environment variable first:

```bash
# From the Gemini provider repository root
export AMPLIFIER_MODULE_PROVIDER_GEMINI=$(pwd)

# Now run with this profile
amplifier run --profile gemini-test "Hello, can you test the Gemini provider?"

# Test with tool usage
amplifier run --profile gemini-test "List files in the current directory"
```

To enable raw debug logging, edit this profile and set `raw_debug: true` in the provider config.

## Configuration

- **Provider**: Local Gemini provider (file://.)
- **Model**: gemini-2.5-flash (balanced performance)
- **Debug**: Enabled (debug=true) for detailed logging
- **Raw Debug**: Disabled by default (set to true for ultra-verbose API I/O)

## Features Being Tested

1. **Three-level debug logging**:
   - INFO: Summary events (always enabled)
   - DEBUG: Truncated request/response (debug=true)
   - RAW: Complete untruncated data (raw_debug=true)

2. **Tool result validation**:
   - Automatic detection of missing tool results
   - Synthetic error injection for graceful recovery
   - Observable via provider:tool_sequence_repaired events

3. **Usage tracking**:
   - Proper Usage model integration
   - Token counting (input/output/total)

4. **Tool calling**:
   - Bash and filesystem tools available for testing
   - Synthetic tool call IDs (Gemini doesn't provide them)

## Testing Scenarios

### Basic Completion
```bash
amplifier run --profile gemini-test "What is 2+2?"
```

### Tool Usage
```bash
amplifier run --profile gemini-test "List files in the current directory"
```

### Thinking/Reasoning
Edit the profile to use `gemini-2.5-pro` and add thinking config for complex reasoning tasks.

## Debug Logs

When debug=true, look for these events in logs:
- `llm:request` - Request summary
- `llm:request:debug` - Detailed request with truncated values
- `llm:response` - Response summary
- `llm:response:debug` - Detailed response with truncated values

When raw_debug=true, also see:
- `llm:request:raw` - Complete untruncated request
- `llm:response:raw` - Complete untruncated response

## Environment Variables

Ensure you have:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).
