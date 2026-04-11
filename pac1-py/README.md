# BitGN PAC1 Python Sample

Runnable Python sample for the `bitgn/pac1-dev` benchmark.

This sample follows the same control-plane flow as the sandbox demo, but it targets the PCM runtime (`bitgn.vm.pcm`) instead of the mini sandbox VM.
The agent loop uses OpenAI `Responses API`, which is the recommended path for GPT-5 models.

You will need to provide either `OPENAI_API_KEY` for standard API auth or
`OPENAI_ACCESS_TOKEN` if you already have a bearer token from another auth layer.
By default, the sample first looks in Windows Credential Manager and only falls
back to `.env` or process environment variables if no stored credential is found.
This sample does not run a browser-based "Sign in with OpenAI" flow by itself.

Quick start:

1. Store your OpenAI secret in Windows Credential Manager
2. Optionally export `BITGN_HOST`, `BENCH_ID`, `MODEL_ID`, `PAC1_LOG_FILE`, or `PAC1_MAX_WORKERS`
3. Run `make sync`
4. Run `make run`

Store the credential under the default target:

```powershell
cmdkey /generic:pac1-py-openai /user:openai /pass:YOUR_OPENAI_SECRET
```

You can override the target name with `OPENAI_CREDENTIAL_TARGET`.

Run it with:

```bash
uv run python main.py
```

Useful environment overrides:

- `BITGN_HOST` defaults to `https://api.bitgn.com`
- `BENCH_ID` defaults to `bitgn/pac1-dev`
- `MODEL_ID` defaults to `gpt-5`
- `OPENAI_REASONING_EFFORT` defaults to `medium` for GPT-5 task steps and `minimal` for the startup preflight
- `OPENAI_CREDENTIAL_TARGET` defaults to `pac1-py-openai`
- Windows Credential Manager is checked before `OPENAI_ACCESS_TOKEN` and `OPENAI_API_KEY`
- `PAC1_LOG_FILE` defaults to `logs/pac1-YYYYMMDD-HHMMSS.log`
- `PAC1_MAX_WORKERS` defaults to `4`
