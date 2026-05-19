# storyboard-cli

Gemini video-to-text storyboard CLI.

`storyboard` takes one local video file and prints one normalized JSON object with a reusable recreation brief.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/radjathaher/storyboard-cli/main/scripts/install.sh | sh
```

Or download a release asset from GitHub Releases.

## Usage

```sh
export GEMINI_API_KEY="..."

storyboard ./video.mp4
storyboard ./video.mp4 --prompt "Focus on wardrobe and camera movement."
storyboard ./video.mp4 \
  --model gemini-3.1-pro-preview \
  --media-resolution high \
  --fps 2 \
  --max-output-tokens 4096
```

## CLI contract

- single root command, no subcommands
- exactly one local video file
- auth: `GEMINI_API_KEY` env only
- stdout: JSON only
- stderr: minimal progress, elapsed time, estimated cost

Defaults:

| flag | default |
|---|---|
| `--model` | `gemini-3.1-pro-preview` |
| `--media-resolution` | `high` |
| `--fps` | `2` |
| `--max-output-tokens` | `4096` |

`--media-resolution`: `low`, `medium`, `high`, `unspecified`.

## Output

```json
{
  "description": "single markdown/text reconstruction brief from Gemini",
  "usage": {
    "model": "gemini-3.1-pro-preview",
    "prompt_token_count": 123,
    "candidates_token_count": 456,
    "thoughts_token_count": 0,
    "total_token_count": 579
  },
  "cost": {
    "currency": "USD",
    "input_usd": 0.001,
    "output_usd": 0.005,
    "total_usd": 0.006,
    "pricing_source": "https://ai.google.dev/gemini-api/docs/pricing"
  },
  "elapsed_seconds": 8.2
}
```

Unknown model pricing keeps `usage`, sets `cost` to `null`, and warns on stderr.

## Behavior

```text
video file
├─ detect MIME
├─ Gemini Files API resumable upload
├─ poll file state every 1s, timeout 5m
├─ generateContent
├─ best-effort delete uploaded file
└─ print normalized JSON
```

No async runtime. v1 is intentionally single-file only. Future batch mode can use blocking worker threads with `--jobs N`.

## Development

```sh
cargo test
cargo build --release
```
