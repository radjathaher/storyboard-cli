use std::{
    env, fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const DEFAULT_MODEL: &str = "google/gemini-2.5-flash";
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;
const DEFAULT_TEMPERATURE: f64 = 1.0;
const DEFAULT_QUEUE_URL: &str = "https://queue.fal.run";
const DEFAULT_REST_URL: &str = "https://rest.fal.ai";
const DEFAULT_CDN_URL: &str = "https://v3.fal.media";
const DEFAULT_CDN_FALLBACK_URL: &str = "https://fal.media";
const ENDPOINT: &str = "openrouter/router/video";
const MULTIPART_THRESHOLD: u64 = 100 * 1024 * 1024;
const MULTIPART_CHUNK_SIZE: u64 = 10 * 1024 * 1024;

#[derive(Debug, Parser)]
#[command(
    name = "storyboard",
    version,
    about = "Describe one video as a reusable storyboard brief through fal OpenRouter video.",
    disable_help_flag = true,
    long_about = r#"Describe one video as a reusable storyboard brief through fal's openrouter/router/video endpoint.

The CLI accepts a local video file, public video URL, or public YouTube URL. Local files are uploaded to fal storage first. The selected Gemini-capable OpenRouter model analyzes the video and the CLI prints provider-shaped JSON on stdout. Progress and cost hints go to stderr."#,
    after_help = r#"Examples:
  storyboard ./video.mp4
  storyboard https://example.com/video.mp4
  storyboard https://www.youtube.com/watch?v=dQw4w9WgXcQ
  storyboard ./video.mp4 --model google/gemini-3.1-pro-preview
  storyboard ./video.mp4 --prompt "Focus on wardrobe and camera movement."

Auth:
  Set FAL_KEY in the environment. The CLI also falls back to /run/secrets/FAL_KEY.

Output contract:
  stdout is JSON only: { provider, endpoint, request_id, model, input, output, usage, elapsed_seconds }
  stderr is human progress only: upload, queue, polling, elapsed/cost summary.

Model guidance:
  Default is google/gemini-2.5-flash for low-cost storyboard and QA runs.
  Use --model google/gemini-3.1-pro-preview when you need deeper visual reasoning.

Pricing guidance:
  fal bills this route by input/output tokens and returns authoritative cost in usage.cost.
  Gemini video is tokenized at roughly 300 tokens/sec at default media resolution, or roughly 100 tokens/sec at low media resolution in Google's direct API docs. The fal route does not expose media_resolution/fps controls.

Docs:
  fal route: https://fal.ai/models/openrouter/router/video/api
  Gemini video: https://ai.google.dev/gemini-api/docs/video-understanding
  OpenRouter models: https://openrouter.ai/api/v1/models"#
)]
struct Cli {
    /// Local video path, public video URL, or public YouTube URL.
    #[arg(
        value_name = "INPUT",
        long_help = "Exactly one input. This can be a local video file (.mp4, .mov, .webm, .mpeg), a public video URL, or a public YouTube URL. Local files are uploaded to fal storage; URLs are passed directly to openrouter/router/video."
    )]
    input: String,

    /// OpenRouter model name routed through fal.
    #[arg(long, default_value = DEFAULT_MODEL, long_help = "OpenRouter model id. Default is google/gemini-2.5-flash for low-cost storyboard/QA. Use google/gemini-3.1-pro-preview for higher quality at materially higher cost.")]
    model: String,

    /// Maximum output tokens.
    #[arg(long, default_value_t = DEFAULT_MAX_OUTPUT_TOKENS, long_help = "Maximum response tokens. Raise this for long videos or very detailed timestamped briefs. Lower it for faster/cheaper summaries.")]
    max_output_tokens: u32,

    /// Sampling temperature passed to fal/OpenRouter.
    #[arg(long, default_value_t = DEFAULT_TEMPERATURE)]
    temperature: f64,

    /// Extra instruction appended to the default recreation prompt.
    #[arg(
        long,
        long_help = r#"Extra instruction appended to the built-in recreation prompt. Example: --prompt "Focus on wardrobe, camera movement, captions, and edit pacing.""#
    )]
    prompt: Option<String>,

    /// Queue poll interval in seconds.
    #[arg(long, default_value_t = 5)]
    poll_interval_secs: u64,

    /// Queue wait timeout in seconds.
    #[arg(long, default_value_t = 1200)]
    max_wait_secs: u64,

    #[arg(short = 'h', long = "help", action = clap::ArgAction::Help, help = "Print help")]
    help: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct InputInfo {
    kind: String,
    source: String,
    resolved_url: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct StoryboardOutput {
    provider: String,
    endpoint: String,
    request_id: String,
    model: String,
    input: InputInfo,
    output: String,
    usage: Value,
    elapsed_seconds: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct QueueStatus {
    status: String,
    request_id: String,
    #[serde(default)]
    response_url: Option<String>,
    #[serde(default)]
    status_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CdnToken {
    token: String,
    token_type: String,
    base_url: String,
}

struct FalClient {
    http: Client,
    fal_key: String,
    queue_url: String,
    rest_url: String,
    cdn_url: String,
    cdn_fallback_url: String,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    validate_cli(&cli)?;
    let started = Instant::now();
    let client = FalClient::new()?;
    let output = generate_storyboard(&client, &cli, started)?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    eprintln!(
        "done in {:.3}s{}",
        output.elapsed_seconds,
        cost_summary(&output.usage)
    );
    Ok(())
}

fn generate_storyboard(
    client: &FalClient,
    cli: &Cli,
    started: Instant,
) -> Result<StoryboardOutput> {
    let input = resolve_input(client, &cli.input)?;
    eprintln!("generating storyboard with {}", cli.model);
    let body = json!({
        "video_urls": [input.resolved_url],
        "prompt": build_prompt(cli.prompt.as_deref()),
        "model": cli.model,
        "temperature": cli.temperature,
        "max_tokens": cli.max_output_tokens,
    });
    let queued = client.submit(ENDPOINT, body)?;
    eprintln!("queued {}; polling", queued.request_id);
    let result = client.wait_result(
        ENDPOINT,
        &queued.request_id,
        cli.max_wait_secs,
        cli.poll_interval_secs,
    )?;
    let output = result
        .get("output")
        .and_then(Value::as_str)
        .context("fal result missing output")?
        .to_string();
    let usage = result.get("usage").cloned().unwrap_or_else(|| json!({}));
    Ok(StoryboardOutput {
        provider: "fal-openrouter-video".to_string(),
        endpoint: ENDPOINT.to_string(),
        request_id: queued.request_id,
        model: cli.model.clone(),
        input,
        output,
        usage,
        elapsed_seconds: round3(started.elapsed().as_secs_f64()),
    })
}

fn validate_cli(cli: &Cli) -> Result<()> {
    if cli.input.trim().is_empty() {
        bail!("input is required");
    }
    if cli.max_output_tokens == 0 {
        bail!("--max-output-tokens must be greater than 0");
    }
    if !(0.0..=2.0).contains(&cli.temperature) || !cli.temperature.is_finite() {
        bail!("--temperature must be between 0 and 2");
    }
    if !is_url(&cli.input) && !Path::new(&cli.input).is_file() {
        bail!("input must be a local file or http(s) URL: {}", cli.input);
    }
    Ok(())
}

fn resolve_input(client: &FalClient, input: &str) -> Result<InputInfo> {
    if is_url(input) {
        return Ok(InputInfo {
            kind: "url".to_string(),
            source: input.to_string(),
            resolved_url: input.to_string(),
        });
    }
    let path = PathBuf::from(input);
    let mime_type = mime_for_path(&path);
    eprintln!("uploading {} ({mime_type})", path.display());
    let resolved_url = client.upload_file(&path)?;
    Ok(InputInfo {
        kind: "file".to_string(),
        source: input.to_string(),
        resolved_url,
    })
}

impl FalClient {
    fn new() -> Result<Self> {
        Self::with_key_and_urls(
            read_secret("FAL_KEY")?,
            env::var("FAL_QUEUE_URL").unwrap_or_else(|_| DEFAULT_QUEUE_URL.to_string()),
            env::var("FAL_REST_URL").unwrap_or_else(|_| DEFAULT_REST_URL.to_string()),
            env::var("FAL_CDN_URL").unwrap_or_else(|_| DEFAULT_CDN_URL.to_string()),
            env::var("FAL_CDN_FALLBACK_URL")
                .unwrap_or_else(|_| DEFAULT_CDN_FALLBACK_URL.to_string()),
        )
    }

    fn with_key_and_urls(
        fal_key: String,
        queue_url: String,
        rest_url: String,
        cdn_url: String,
        cdn_fallback_url: String,
    ) -> Result<Self> {
        Ok(Self {
            http: Client::builder()
                .user_agent(concat!("storyboard-cli/", env!("CARGO_PKG_VERSION")))
                .timeout(Duration::from_secs(120))
                .build()?,
            fal_key,
            queue_url,
            rest_url,
            cdn_url,
            cdn_fallback_url,
        })
    }

    fn auth_header(&self) -> String {
        format!("Key {}", self.fal_key)
    }

    fn bearer_header(&self) -> String {
        format!("Bearer {}", self.fal_key)
    }

    fn cdn_token(&self) -> Result<CdnToken> {
        let url = format!(
            "{}/storage/auth/token?storage_type=fal-cdn-v3",
            self.rest_url.trim_end_matches('/')
        );
        let res = self
            .http
            .post(url)
            .header("Authorization", self.auth_header())
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&json!({}))
            .send()?;
        decode(res).and_then(|v| serde_json::from_value(v).context("decode CDN token"))
    }

    fn upload_file(&self, path: &Path) -> Result<String> {
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let name = path
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or("upload.bin")
            .to_string();
        let content_type = mime_for_path(path);
        let mut errors = Vec::new();
        match self.upload_v3(&bytes, &name, content_type, path) {
            Ok(url) => return Ok(url),
            Err(err) => errors.push(format!("fal_v3: {err:#}")),
        }
        match self.upload_cdn(&bytes, &name, content_type) {
            Ok(url) => return Ok(url),
            Err(err) => errors.push(format!("cdn: {err:#}")),
        }
        match self.upload_storage(&bytes, &name, content_type) {
            Ok(url) => return Ok(url),
            Err(err) => errors.push(format!("storage: {err:#}")),
        }
        bail!("all fal upload methods failed: {}", errors.join(" | "))
    }

    fn upload_v3(
        &self,
        bytes: &[u8],
        name: &str,
        content_type: &str,
        path: &Path,
    ) -> Result<String> {
        let token = self.cdn_token()?;
        if fs::metadata(path)?.len() > MULTIPART_THRESHOLD {
            return self.upload_v3_multipart(path, name, content_type, &token);
        }
        let url = format!("{}/files/upload", self.cdn_url.trim_end_matches('/'));
        let res = self
            .http
            .post(url)
            .header(
                "Authorization",
                format!("{} {}", token.token_type, token.token),
            )
            .header("Content-Type", content_type)
            .header("X-Fal-File-Name", name)
            .body(bytes.to_vec())
            .send()?;
        decode(res)?
            .get("access_url")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .context("v3 upload missing access_url")
    }

    fn upload_v3_multipart(
        &self,
        path: &Path,
        name: &str,
        content_type: &str,
        token: &CdnToken,
    ) -> Result<String> {
        let auth = format!("{} {}", token.token_type, token.token);
        let create_url = format!(
            "{}/files/upload/multipart",
            token.base_url.trim_end_matches('/')
        );
        let created = decode(
            self.http
                .post(create_url)
                .header("Authorization", &auth)
                .header("Accept", "application/json")
                .header("Content-Type", content_type)
                .header("X-Fal-File-Name", name)
                .send()?,
        )?;
        let access_url = created
            .get("access_url")
            .and_then(Value::as_str)
            .context("multipart missing access_url")?
            .to_string();
        let upload_id = created
            .get("uploadId")
            .and_then(Value::as_str)
            .context("multipart missing uploadId")?
            .to_string();
        let size = fs::metadata(path)?.len();
        let parts = size.div_ceil(MULTIPART_CHUNK_SIZE);
        let mut file = fs::File::open(path)?;
        let mut part_json = Vec::new();
        for part in 1..=parts {
            let start = (part - 1) * MULTIPART_CHUNK_SIZE;
            file.seek(SeekFrom::Start(start))?;
            let chunk_len = MULTIPART_CHUNK_SIZE.min(size - start) as usize;
            let mut buf = vec![0u8; chunk_len];
            file.read_exact(&mut buf)?;
            let part_url = format!(
                "{}/multipart/{}/{}",
                access_url.trim_end_matches('/'),
                upload_id,
                part
            );
            let res = self
                .http
                .put(part_url)
                .header("Authorization", &auth)
                .header("Content-Type", content_type)
                .header("Accept-Encoding", "identity")
                .body(buf)
                .send()?;
            if !res.status().is_success() {
                bail!("multipart part {} failed: http {}", part, res.status());
            }
            let etag = res
                .headers()
                .get("etag")
                .and_then(|v| v.to_str().ok())
                .context("multipart part missing etag")?
                .to_string();
            part_json.push(json!({"partNumber": part, "etag": etag}));
        }
        let complete_url = format!(
            "{}/multipart/{}/complete",
            access_url.trim_end_matches('/'),
            upload_id
        );
        decode(
            self.http
                .post(complete_url)
                .header("Authorization", auth)
                .json(&json!({"parts": part_json}))
                .send()?,
        )?;
        Ok(access_url)
    }

    fn upload_cdn(&self, bytes: &[u8], name: &str, content_type: &str) -> Result<String> {
        let url = format!(
            "{}/files/upload",
            self.cdn_fallback_url.trim_end_matches('/')
        );
        let res = self
            .http
            .post(url)
            .header("Authorization", self.bearer_header())
            .header("Content-Type", content_type)
            .header("X-Fal-File-Name", name)
            .body(bytes.to_vec())
            .send()?;
        decode(res)?
            .get("access_url")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .context("cdn upload missing access_url")
    }

    fn upload_storage(&self, bytes: &[u8], name: &str, content_type: &str) -> Result<String> {
        let init_url = format!(
            "{}/storage/upload/initiate?storage_type=gcs",
            self.rest_url.trim_end_matches('/')
        );
        let init = decode(
            self.http
                .post(init_url)
                .header("Authorization", self.auth_header())
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .json(&json!({"file_name": name, "content_type": content_type}))
                .send()?,
        )?;
        let upload_url = init
            .get("upload_url")
            .and_then(Value::as_str)
            .context("storage missing upload_url")?;
        let file_url = init
            .get("file_url")
            .and_then(Value::as_str)
            .context("storage missing file_url")?
            .to_string();
        let res = self
            .http
            .put(upload_url)
            .header("Content-Type", content_type)
            .body(bytes.to_vec())
            .send()?;
        if !res.status().is_success() {
            bail!("storage PUT failed: http {}", res.status());
        }
        Ok(file_url)
    }

    fn submit(&self, endpoint: &str, body: Value) -> Result<QueueStatus> {
        let url = format!(
            "{}/{}",
            self.queue_url.trim_end_matches('/'),
            endpoint.trim_matches('/')
        );
        let res = self
            .http
            .post(url)
            .header("Authorization", self.auth_header())
            .json(&body)
            .send()?;
        decode(res).and_then(|v| serde_json::from_value(v).context("decode queue status"))
    }

    fn get_status(&self, endpoint: &str, request_id: &str) -> Result<Value> {
        let url = format!(
            "{}/{}/requests/{}/status",
            self.queue_url.trim_end_matches('/'),
            endpoint.trim_matches('/'),
            request_id
        );
        decode(
            self.http
                .get(url)
                .header("Authorization", self.auth_header())
                .send()?,
        )
    }

    fn get_result(&self, endpoint: &str, request_id: &str) -> Result<Value> {
        let url = format!(
            "{}/{}/requests/{}",
            self.queue_url.trim_end_matches('/'),
            endpoint.trim_matches('/'),
            request_id
        );
        decode(
            self.http
                .get(url)
                .header("Authorization", self.auth_header())
                .send()?,
        )
    }

    fn wait_result(
        &self,
        endpoint: &str,
        request_id: &str,
        max_wait_secs: u64,
        poll_interval_secs: u64,
    ) -> Result<Value> {
        let start = Instant::now();
        loop {
            let status = self.get_status(endpoint, request_id)?;
            match status.get("status").and_then(Value::as_str) {
                Some("COMPLETED") => return self.get_result(endpoint, request_id),
                Some("FAILED") => bail!("fal task failed: {}", status),
                _ => {
                    if start.elapsed() > Duration::from_secs(max_wait_secs) {
                        bail!("timeout waiting for {request_id}");
                    }
                    thread::sleep(Duration::from_secs(poll_interval_secs.max(1)));
                }
            }
        }
    }
}

fn decode(res: reqwest::blocking::Response) -> Result<Value> {
    let status = res.status();
    let text = res.text()?;
    let parsed = serde_json::from_str::<Value>(&text).unwrap_or_else(|_| json!({"raw": text}));
    if !status.is_success() {
        bail!("http {status}: {parsed}");
    }
    Ok(parsed)
}

fn read_secret(name: &str) -> Result<String> {
    if let Ok(v) = env::var(name) {
        if !v.trim().is_empty() {
            return Ok(v.trim().to_string());
        }
    }
    let p = format!("/run/secrets/{name}");
    if let Ok(v) = fs::read_to_string(&p) {
        if !v.trim().is_empty() {
            return Ok(v.trim().to_string());
        }
    }
    bail!("{name} missing; set ${name} or /run/secrets/{name}")
}

fn build_prompt(extra: Option<&str>) -> String {
    let mut prompt = String::from(
        "Create a detailed video recreation brief from this video. Return only the brief text, not JSON. Include scene/timestamp notes, camera framing and motion, subjects and actions, on-screen text, audio or spoken content, pacing, transitions, visual style, and enough concrete detail to recreate the same video in another AI video tool.",
    );
    if let Some(extra) = extra.filter(|value| !value.trim().is_empty()) {
        prompt.push_str("\n\nExtra instruction: ");
        prompt.push_str(extra.trim());
    }
    prompt
}

fn mime_for_path(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|v| v.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "mp4" | "m4v" => "video/mp4",
        "mpeg" | "mpg" => "video/mpeg",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        _ => "application/octet-stream",
    }
}

fn is_url(input: &str) -> bool {
    input.starts_with("http://") || input.starts_with("https://")
}

fn cost_summary(usage: &Value) -> String {
    usage
        .get("cost")
        .and_then(Value::as_f64)
        .map(|cost| format!("; provider cost ${cost:.6}"))
        .unwrap_or_else(|| "; provider cost unavailable".to_string())
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use httpmock::prelude::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn test_client(server: &MockServer) -> FalClient {
        FalClient::with_key_and_urls(
            "test-key".to_string(),
            server.base_url(),
            server.base_url(),
            server.base_url(),
            server.base_url(),
        )
        .unwrap()
    }

    #[test]
    fn cli_defaults_are_public_contract() {
        let cli = Cli::try_parse_from(["storyboard", "https://youtu.be/example"]).unwrap();
        assert_eq!(cli.model, DEFAULT_MODEL);
        assert_eq!(cli.max_output_tokens, 4096);
        assert_eq!(cli.temperature, 1.0);
    }

    #[test]
    fn validates_local_file_or_url_input() {
        let cli = Cli::try_parse_from(["storyboard", "https://example.com/video.mp4"]).unwrap();
        validate_cli(&cli).unwrap();

        let cli = Cli::try_parse_from(["storyboard", "/definitely/missing.mp4"]).unwrap();
        assert!(validate_cli(&cli).is_err());
    }

    #[test]
    fn mime_fallbacks_cover_common_video_extensions() {
        for (suffix, mime) in [
            (".mp4", "video/mp4"),
            (".mov", "video/quicktime"),
            (".webm", "video/webm"),
            (".mpeg", "video/mpeg"),
        ] {
            let file = tempfile::Builder::new().suffix(suffix).tempfile().unwrap();
            assert_eq!(mime_for_path(file.path()), mime);
        }
    }

    #[test]
    fn prompt_appends_extra_instruction() {
        let prompt = build_prompt(Some("Focus on racket motion."));
        assert!(prompt.contains("video recreation brief"));
        assert!(prompt.contains("Extra instruction: Focus on racket motion."));
    }

    #[test]
    fn help_explains_fal_openrouter_and_pricing_shape() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("fal's openrouter/router/video"));
        assert!(help.contains("FAL_KEY"));
        assert!(help.contains("usage.cost"));
        assert!(help.contains("google/gemini-2.5-flash"));
    }

    #[test]
    fn direct_url_flow_outputs_provider_json() {
        let server = MockServer::start();
        let submit = server.mock(|when, then| {
            when.method(POST)
                .path("/openrouter/router/video")
                .header("authorization", "Key test-key")
                .json_body_partial(r#"{"video_urls":["https://youtu.be/example"]}"#);
            then.status(200).json_body(json!({
                "status": "IN_QUEUE",
                "request_id": "req-url"
            }));
        });
        let poll = server.mock(|when, then| {
            when.method(GET)
                .path("/openrouter/router/video/requests/req-url/status");
            then.status(200).json_body(json!({
                "status": "COMPLETED",
                "request_id": "req-url"
            }));
        });
        let result = server.mock(|when, then| {
            when.method(GET).path("/openrouter/router/video/requests/req-url");
            then.status(200).json_body(json!({
                "output": "Shot-by-shot brief.",
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100, "cost": 0.0005}
            }));
        });

        let cli = Cli::try_parse_from([
            "storyboard",
            "https://youtu.be/example",
            "--poll-interval-secs",
            "1",
            "--max-wait-secs",
            "5",
        ])
        .unwrap();
        let out = generate_storyboard(&test_client(&server), &cli, Instant::now()).unwrap();
        assert_eq!(out.provider, "fal-openrouter-video");
        assert_eq!(out.endpoint, ENDPOINT);
        assert_eq!(out.request_id, "req-url");
        assert_eq!(out.input.kind, "url");
        assert_eq!(out.output, "Shot-by-shot brief.");
        assert_eq!(out.usage.get("cost").and_then(Value::as_f64), Some(0.0005));
        submit.assert();
        poll.assert();
        result.assert();
    }

    #[test]
    fn local_file_uploads_before_queue_submit() {
        let server = MockServer::start();
        let token = server.mock(|when, then| {
            when.method(POST)
                .path("/storage/auth/token")
                .query_param("storage_type", "fal-cdn-v3");
            then.status(200).json_body(json!({
                "token": "upload-token",
                "token_type": "Bearer",
                "base_url": server.base_url()
            }));
        });
        let upload = server.mock(|when, then| {
            when.method(POST)
                .path("/files/upload")
                .header("authorization", "Bearer upload-token")
                .header("content-type", "video/mp4");
            then.status(200).json_body(json!({
                "access_url": "https://files.example/input.mp4"
            }));
        });
        let submit = server.mock(|when, then| {
            when.method(POST)
                .path("/openrouter/router/video")
                .json_body_partial(r#"{"video_urls":["https://files.example/input.mp4"]}"#);
            then.status(200).json_body(json!({
                "status": "IN_QUEUE",
                "request_id": "req-file"
            }));
        });
        server.mock(|when, then| {
            when.method(GET)
                .path("/openrouter/router/video/requests/req-file/status");
            then.status(200)
                .json_body(json!({"status": "COMPLETED", "request_id": "req-file"}));
        });
        server.mock(|when, then| {
            when.method(GET)
                .path("/openrouter/router/video/requests/req-file");
            then.status(200).json_body(json!({
                "output": "Uploaded file brief.",
                "usage": {"cost": 0.001}
            }));
        });

        let mut file = NamedTempFile::with_suffix(".mp4").unwrap();
        file.write_all(b"fake mp4").unwrap();
        let cli = Cli::try_parse_from(["storyboard", file.path().to_str().unwrap()]).unwrap();
        let out = generate_storyboard(&test_client(&server), &cli, Instant::now()).unwrap();
        assert_eq!(out.input.kind, "file");
        assert_eq!(out.input.resolved_url, "https://files.example/input.mp4");
        assert_eq!(out.output, "Uploaded file brief.");
        token.assert();
        upload.assert();
        submit.assert();
    }

    #[test]
    fn failed_queue_status_is_clear() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/openrouter/router/video");
            then.status(200)
                .json_body(json!({"status": "IN_QUEUE", "request_id": "req-fail"}));
        });
        server.mock(|when, then| {
            when.method(GET)
                .path("/openrouter/router/video/requests/req-fail/status");
            then.status(200)
                .json_body(json!({"status": "FAILED", "detail": "bad input"}));
        });
        let cli = Cli::try_parse_from(["storyboard", "https://example.com/video.mp4"]).unwrap();
        let err = generate_storyboard(&test_client(&server), &cli, Instant::now()).unwrap_err();
        assert!(format!("{err:#}").contains("fal task failed"));
    }
}
