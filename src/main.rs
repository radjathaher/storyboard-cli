use std::{
    env, fs,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use reqwest::blocking::Client;
use reqwest::header::{CONTENT_LENGTH, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const DEFAULT_MODEL: &str = "gemini-3.1-pro-preview";
const DEFAULT_MEDIA_RESOLUTION: MediaResolution = MediaResolution::High;
const DEFAULT_FPS: f64 = 2.0;
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;
const PRICING_SOURCE: &str = "https://ai.google.dev/gemini-api/docs/pricing";
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";

#[derive(Debug, Parser)]
#[command(
    name = "storyboard",
    version,
    about = "Describe a local video as a reusable Gemini storyboard brief"
)]
struct Cli {
    /// Local video file to describe.
    video: PathBuf,

    /// Gemini model name.
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,

    /// Gemini media resolution hint.
    #[arg(long, value_enum, default_value_t = DEFAULT_MEDIA_RESOLUTION)]
    media_resolution: MediaResolution,

    /// Video sampling FPS passed to Gemini.
    #[arg(long, default_value_t = DEFAULT_FPS)]
    fps: f64,

    /// Maximum output tokens.
    #[arg(long, default_value_t = DEFAULT_MAX_OUTPUT_TOKENS)]
    max_output_tokens: u32,

    /// Extra instruction appended to the default recreation prompt.
    #[arg(long)]
    prompt: Option<String>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MediaResolution {
    Low,
    Medium,
    High,
    Unspecified,
}

impl std::fmt::Display for MediaResolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Unspecified => "unspecified",
        })
    }
}

impl MediaResolution {
    fn api_value(self) -> &'static str {
        match self {
            Self::Low => "MEDIA_RESOLUTION_LOW",
            Self::Medium => "MEDIA_RESOLUTION_MEDIUM",
            Self::High => "MEDIA_RESOLUTION_HIGH",
            Self::Unspecified => "MEDIA_RESOLUTION_UNSPECIFIED",
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct StoryboardOutput {
    description: String,
    usage: Usage,
    cost: Option<Cost>,
    elapsed_seconds: f64,
}

#[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
struct Usage {
    model: String,
    prompt_token_count: u64,
    candidates_token_count: u64,
    thoughts_token_count: u64,
    total_token_count: u64,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Cost {
    currency: String,
    input_usd: f64,
    output_usd: f64,
    total_usd: f64,
    pricing_source: String,
}

#[derive(Debug, Deserialize)]
struct FileResource {
    name: String,
    uri: String,
    #[serde(default, rename = "mimeType")]
    mime_type: Option<String>,
    #[serde(default)]
    state: Option<String>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let started = Instant::now();
    let api_key = env::var("GEMINI_API_KEY").context("GEMINI_API_KEY is required")?;
    validate_cli(&cli)?;

    let mime_type = detect_mime(&cli.video)?;
    eprintln!("uploading {} ({mime_type})", cli.video.display());

    let client = GeminiClient::new(api_key)?;
    let uploaded = client.upload_file(&cli.video, &mime_type)?;
    eprintln!("uploaded {}; waiting for ACTIVE", uploaded.name);

    let result = (|| -> Result<StoryboardOutput> {
        let file = client.wait_active(&uploaded.name, Duration::from_secs(300))?;
        eprintln!("generating storyboard with {}", cli.model);
        let response = client.generate(&cli, &file)?;
        let description = extract_description(&response)?;
        let usage = extract_usage(&cli.model, &response);
        let cost = estimate_cost(&usage);
        if cost.is_none() {
            eprintln!(
                "warning: no baked pricing for model {}; cost set to null",
                usage.model
            );
        }
        Ok(StoryboardOutput {
            description,
            usage,
            cost,
            elapsed_seconds: round3(started.elapsed().as_secs_f64()),
        })
    })();

    if let Err(err) = client.delete_file(&uploaded.name) {
        eprintln!(
            "warning: failed to delete uploaded file {}: {err:#}",
            uploaded.name
        );
    } else {
        eprintln!("deleted uploaded file {}", uploaded.name);
    }

    let output = result?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    eprintln!(
        "done in {:.3}s{}",
        output.elapsed_seconds,
        cost_summary(output.cost.as_ref())
    );
    Ok(())
}

fn validate_cli(cli: &Cli) -> Result<()> {
    if !cli.video.is_file() {
        bail!("video path must be a local file: {}", cli.video.display());
    }
    if cli.fps <= 0.0 || !cli.fps.is_finite() {
        bail!("--fps must be a positive number");
    }
    if cli.max_output_tokens == 0 {
        bail!("--max-output-tokens must be greater than 0");
    }
    Ok(())
}

fn detect_mime(path: &Path) -> Result<String> {
    if let Some(kind) = infer::get_from_path(path).context("failed to inspect file type")? {
        return Ok(kind.mime_type().to_string());
    }

    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "mp4" | "m4v" => Ok("video/mp4".to_string()),
        "mov" => Ok("video/quicktime".to_string()),
        "webm" => Ok("video/webm".to_string()),
        _ => bail!("could not detect video MIME type for {}", path.display()),
    }
}

struct GeminiClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl GeminiClient {
    fn new(api_key: String) -> Result<Self> {
        let base_url =
            env::var("GEMINI_API_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        Self::with_base_url(api_key, base_url)
    }

    fn with_base_url(api_key: String, base_url: String) -> Result<Self> {
        Ok(Self {
            client: Client::builder()
                .timeout(Duration::from_secs(120))
                .build()?,
            api_key,
            base_url,
        })
    }

    fn upload_file(&self, path: &Path, mime_type: &str) -> Result<FileResource> {
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let display_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("video");
        let start_url = format!("{}/upload/v1beta/files?key={}", self.base_url, self.api_key);
        let metadata = json!({ "file": { "display_name": display_name } });

        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Goog-Upload-Protocol",
            HeaderValue::from_static("resumable"),
        );
        headers.insert("X-Goog-Upload-Command", HeaderValue::from_static("start"));
        headers.insert(
            "X-Goog-Upload-Header-Content-Type",
            HeaderValue::from_str(mime_type)?,
        );
        headers.insert(
            "X-Goog-Upload-Header-Content-Length",
            HeaderValue::from_str(&bytes.len().to_string())?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let start = self
            .client
            .post(start_url)
            .headers(headers)
            .json(&metadata)
            .send()
            .context("failed to start resumable upload")?;
        let start = ensure_success(start, "upload start")?;
        let upload_url = start
            .headers()
            .get("X-Goog-Upload-URL")
            .ok_or_else(|| anyhow!("upload start response missing X-Goog-Upload-URL"))?
            .to_str()?
            .to_string();

        let finalize = self
            .client
            .post(upload_url)
            .header("X-Goog-Upload-Offset", "0")
            .header("X-Goog-Upload-Command", "upload, finalize")
            .header(CONTENT_LENGTH, bytes.len())
            .header(CONTENT_TYPE, mime_type)
            .body(bytes)
            .send()
            .context("failed to finalize upload")?;
        let body: Value = ensure_success(finalize, "upload finalize")?
            .json()
            .context("invalid upload finalize JSON")?;
        parse_file(body)
    }

    fn wait_active(&self, name: &str, timeout: Duration) -> Result<FileResource> {
        let deadline = Instant::now() + timeout;
        loop {
            let file = self.get_file(name)?;
            match file.state.as_deref() {
                Some("ACTIVE") => return Ok(file),
                Some("FAILED") => bail!("uploaded file processing failed: {name}"),
                state => eprintln!("file state: {}", state.unwrap_or("UNKNOWN")),
            }
            if Instant::now() >= deadline {
                bail!(
                    "timed out waiting for uploaded file to become ACTIVE after {}s",
                    timeout.as_secs()
                );
            }
            thread::sleep(Duration::from_secs(1));
        }
    }

    fn get_file(&self, name: &str) -> Result<FileResource> {
        let url = format!("{}/v1beta/{}?key={}", self.base_url, name, self.api_key);
        let response = self.client.get(url).send().context("failed to poll file")?;
        let body: Value = ensure_success(response, "file poll")?
            .json()
            .context("invalid file poll JSON")?;
        parse_file(body)
    }

    fn generate(&self, cli: &Cli, file: &FileResource) -> Result<Value> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, cli.model, self.api_key
        );
        let request = json!({
            "contents": [{
                "role": "user",
                "parts": [
                    { "text": build_prompt(cli.prompt.as_deref()) },
                    {
                        "file_data": { "mime_type": file.mime_type.as_deref().unwrap_or("video/mp4"), "file_uri": file.uri },
                        "video_metadata": { "fps": cli.fps }
                    }
                ]
            }],
            "generationConfig": {
                "maxOutputTokens": cli.max_output_tokens,
                "mediaResolution": cli.media_resolution.api_value()
            }
        });
        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .context("failed to call generateContent")?;
        ensure_success(response, "generateContent")?
            .json()
            .context("invalid generateContent JSON")
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        let url = format!("{}/v1beta/{}?key={}", self.base_url, name, self.api_key);
        let response = self
            .client
            .delete(url)
            .send()
            .context("failed to delete file")?;
        ensure_success(response, "file delete")?;
        Ok(())
    }
}

fn ensure_success(
    response: reqwest::blocking::Response,
    context: &str,
) -> Result<reqwest::blocking::Response> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let text = response.text().unwrap_or_default();
    let message = normalize_gemini_error(&text).unwrap_or_else(|| text.trim().to_string());
    bail!("{context} failed with HTTP {status}: {message}");
}

fn parse_file(body: Value) -> Result<FileResource> {
    let value = body.get("file").cloned().unwrap_or(body);
    serde_json::from_value(value).context("invalid Gemini file resource")
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

fn extract_description(response: &Value) -> Result<String> {
    let parts = response
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("generateContent response missing candidate text"))?;
    let text = parts
        .iter()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();
    if text.is_empty() {
        bail!("generateContent returned empty description");
    }
    Ok(text)
}

fn extract_usage(model: &str, response: &Value) -> Usage {
    let metadata = response.get("usageMetadata").unwrap_or(&Value::Null);
    Usage {
        model: model.to_string(),
        prompt_token_count: get_u64(metadata, "promptTokenCount"),
        candidates_token_count: get_u64(metadata, "candidatesTokenCount"),
        thoughts_token_count: get_u64(metadata, "thoughtsTokenCount"),
        total_token_count: get_u64(metadata, "totalTokenCount"),
    }
}

fn get_u64(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn estimate_cost(usage: &Usage) -> Option<Cost> {
    let pricing = pricing_for_model(&usage.model)?;
    let input_usd =
        round6(usage.prompt_token_count as f64 / 1_000_000.0 * pricing.input_per_million);
    let output_tokens = usage.candidates_token_count + usage.thoughts_token_count;
    let output_usd = round6(output_tokens as f64 / 1_000_000.0 * pricing.output_per_million);
    Some(Cost {
        currency: "USD".to_string(),
        input_usd,
        output_usd,
        total_usd: round6(input_usd + output_usd),
        pricing_source: PRICING_SOURCE.to_string(),
    })
}

struct Pricing {
    input_per_million: f64,
    output_per_million: f64,
}

fn pricing_for_model(model: &str) -> Option<Pricing> {
    match model {
        "gemini-3.1-pro-preview" => Some(Pricing {
            input_per_million: 2.0,
            output_per_million: 12.0,
        }),
        "gemini-3-pro-preview" => Some(Pricing {
            input_per_million: 2.0,
            output_per_million: 12.0,
        }),
        "gemini-2.5-pro" => Some(Pricing {
            input_per_million: 1.25,
            output_per_million: 10.0,
        }),
        "gemini-2.5-flash" => Some(Pricing {
            input_per_million: 0.30,
            output_per_million: 2.50,
        }),
        "gemini-2.5-flash-lite" => Some(Pricing {
            input_per_million: 0.10,
            output_per_million: 0.40,
        }),
        _ => None,
    }
}

fn normalize_gemini_error(text: &str) -> Option<String> {
    let value: Value = serde_json::from_str(text).ok()?;
    let error = value.get("error")?;
    let status = error
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("UNKNOWN");
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("Gemini API error");
    Some(format!("{status}: {message}"))
}

fn cost_summary(cost: Option<&Cost>) -> String {
    match cost {
        Some(cost) => format!("; estimated cost ${:.6}", cost.total_usd),
        None => "; estimated cost unavailable".to_string(),
    }
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}
fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn cli_defaults_are_public_contract() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"not a real mp4").unwrap();
        let cli = Cli::try_parse_from(["storyboard", file.path().to_str().unwrap()]).unwrap();
        assert_eq!(cli.model, DEFAULT_MODEL);
        assert_eq!(cli.media_resolution.to_string(), "high");
        assert_eq!(cli.fps, 2.0);
        assert_eq!(cli.max_output_tokens, 4096);
    }

    #[test]
    fn mime_fallbacks_cover_common_video_extensions() {
        for (suffix, mime) in [
            (".mp4", "video/mp4"),
            (".mov", "video/quicktime"),
            (".webm", "video/webm"),
        ] {
            let file = tempfile::Builder::new().suffix(suffix).tempfile().unwrap();
            assert_eq!(detect_mime(file.path()).unwrap(), mime);
        }
    }

    #[test]
    fn pricing_uses_usage_metadata_tokens() {
        let usage = Usage {
            model: DEFAULT_MODEL.to_string(),
            prompt_token_count: 1_000,
            candidates_token_count: 2_000,
            thoughts_token_count: 500,
            total_token_count: 3_500,
        };
        let cost = estimate_cost(&usage).unwrap();
        assert_eq!(cost.input_usd, 0.002);
        assert_eq!(cost.output_usd, 0.03);
        assert_eq!(cost.total_usd, 0.032);
    }

    #[test]
    fn unknown_model_has_no_cost() {
        let usage = Usage {
            model: "custom-model".to_string(),
            ..Usage::default()
        };
        assert!(estimate_cost(&usage).is_none());
    }

    #[test]
    fn gemini_error_normalization_is_clear() {
        let raw =
            r#"{"error":{"code":401,"message":"API key invalid","status":"UNAUTHENTICATED"}}"#;
        assert_eq!(
            normalize_gemini_error(raw).unwrap(),
            "UNAUTHENTICATED: API key invalid"
        );
    }

    #[test]
    fn mocked_gemini_flow_outputs_normalized_json_and_deletes_file() {
        let server = MockServer::start();
        let upload_start = server.mock(|when, then| {
            when.method(POST)
                .path("/upload/v1beta/files")
                .query_param("key", "test-key");
            then.status(200).header(
                "X-Goog-Upload-URL",
                &format!("{}/upload-session", server.base_url()),
            );
        });
        let upload_finalize = server.mock(|when, then| {
            when.method(POST).path("/upload-session");
            then.status(200).json_body(json!({"file":{"name":"files/abc","uri":"https://files.local/abc","state":"PROCESSING"}}));
        });
        let poll = server.mock(|when, then| {
            when.method(GET)
                .path("/v1beta/files/abc")
                .query_param("key", "test-key");
            then.status(200).json_body(
                json!({"name":"files/abc","uri":"https://files.local/abc","state":"ACTIVE"}),
            );
        });
        let generate = server.mock(|when, then| {
            when.method(POST).path(format!("/v1beta/models/{DEFAULT_MODEL}:generateContent")).query_param("key", "test-key");
            then.status(200).json_body(json!({
                "candidates":[{"content":{"parts":[{"text":"A detailed reconstruction brief."}]}}],
                "usageMetadata":{"promptTokenCount":1000,"candidatesTokenCount":2000,"thoughtsTokenCount":0,"totalTokenCount":3000}
            }));
        });
        let delete = server.mock(|when, then| {
            when.method(DELETE)
                .path("/v1beta/files/abc")
                .query_param("key", "test-key");
            then.status(200).json_body(json!({}));
        });

        let file = tempfile::Builder::new().suffix(".mp4").tempfile().unwrap();
        fs::write(file.path(), b"fake mp4").unwrap();

        let cli = Cli::try_parse_from(["storyboard", file.path().to_str().unwrap()]).unwrap();
        let client =
            GeminiClient::with_base_url("test-key".to_string(), server.base_url()).unwrap();
        let mime = detect_mime(file.path()).unwrap();
        let uploaded = client.upload_file(file.path(), &mime).unwrap();
        let active = client
            .wait_active(&uploaded.name, Duration::from_secs(2))
            .unwrap();
        let response = client.generate(&cli, &active).unwrap();
        client.delete_file(&uploaded.name).unwrap();

        let json = StoryboardOutput {
            description: extract_description(&response).unwrap(),
            usage: extract_usage(&cli.model, &response),
            cost: estimate_cost(&extract_usage(&cli.model, &response)),
            elapsed_seconds: 0.0,
        };
        assert_eq!(json.description, "A detailed reconstruction brief.");
        assert_eq!(json.usage.total_token_count, 3000);
        assert_eq!(json.cost.unwrap().total_usd, 0.026);

        upload_start.assert();
        upload_finalize.assert();
        poll.assert();
        generate.assert();
        delete.assert();
    }
}
