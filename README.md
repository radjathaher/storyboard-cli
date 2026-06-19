# storyboard-cli

Video-to-text storyboard CLI powered by fal `openrouter/router/video`.

`storyboard` accepts one local video file, public video URL, or public YouTube URL and prints one provider-shaped JSON object with a reusable recreation brief.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/radjathaher/storyboard-cli/main/scripts/install.sh | sh
```

Or download a release asset from GitHub Releases.

## Usage

```sh
export FAL_KEY="..."

storyboard ./video.mp4
storyboard https://example.com/video.mp4
storyboard https://www.youtube.com/watch?v=dQw4w9WgXcQ
storyboard ./video.mp4 --prompt "Focus on wardrobe and camera movement."
storyboard ./video.mp4 \
  --model google/gemini-3.1-pro-preview \
  --max-output-tokens 4096
```

The CLI also falls back to `/run/secrets/FAL_KEY`.

## CLI contract

- single root command, no subcommands
- exactly one input: local file, public video URL, or public YouTube URL
- auth: `FAL_KEY` env or `/run/secrets/FAL_KEY`
- stdout: JSON only
- stderr: progress, queue status, elapsed time, provider cost summary

Defaults:

| flag | default |
|---|---|
| `--model` | `google/gemini-2.5-flash` |
| `--max-output-tokens` | `4096` |
| `--temperature` | `1` |
| `--poll-interval-secs` | `5` |
| `--max-wait-secs` | `1200` |

## Output

```json
{
  "provider": "fal-openrouter-video",
  "endpoint": "openrouter/router/video",
  "request_id": "req_...",
  "model": "google/gemini-2.5-flash",
  "input": {
    "kind": "file",
    "source": "./video.mp4",
    "resolved_url": "https://v3b.fal.media/files/.../video.mp4"
  },
  "output": "single markdown/text reconstruction brief from the model",
  "usage": {
    "prompt_tokens": 18000,
    "completion_tokens": 1200,
    "total_tokens": 19200,
    "cost": 0.004
  },
  "elapsed_seconds": 8.2
}
```

`usage.cost` is the authoritative provider cost returned by fal/OpenRouter.

## Pricing mental model

Gemini video understanding is token-priced, not priced directly per file or minute. Google documents rough video tokenization at about 300 tokens/sec at default media resolution, or about 100 tokens/sec at low media resolution. The fal route does not expose direct `media_resolution` or `fps` knobs, so treat returned `usage.cost` as truth.

OpenRouter model catalog currently lists:

| model | input | output |
|---|---:|---:|
| `google/gemini-2.5-flash` | `$0.30 / 1M` | `$2.50 / 1M` |
| `google/gemini-3.1-pro-preview` | `$2.00 / 1M` | `$12.00 / 1M` |

Use `google/gemini-2.5-flash` for normal storyboard and QA runs. Use `google/gemini-3.1-pro-preview` when deeper visual reasoning is worth the cost.

## Behavior

```text
input
├─ local file → upload to fal storage
├─ URL/YouTube → pass through directly
├─ submit fal queue request to openrouter/router/video
├─ poll queue until complete
└─ print provider-shaped JSON
```

No async runtime. v1 is intentionally one video only.

## Development

```sh
cargo test
cargo build --release
```
