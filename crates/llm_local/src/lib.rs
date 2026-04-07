use anyhow::{anyhow, Context, Result};
use std::process::Command;

pub trait LlmClient {
    fn answer(&self, system_prompt: &str, user_prompt: &str) -> Result<String>;
}

pub enum Backend {
    Local,
    Bedrock,
}

pub fn build_client(backend: Backend, model: String) -> Box<dyn LlmClient> {
    match backend {
        Backend::Local => Box::new(LocalLlamaCppClient { model }),
        Backend::Bedrock => Box::new(BedrockClient { model }),
    }
}

pub struct LocalLlamaCppClient {
    pub model: String,
}

impl LlmClient for LocalLlamaCppClient {
    fn answer(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let prompt = format!("[SYSTEM]\n{}\n\n[USER]\n{}", system_prompt, user_prompt);

        // Force one-shot completion mode to avoid interactive chat prompts.
        let output = run_llama("llama-completion", &self.model, &prompt)?;

        if !output.status.success() {
            return Err(anyhow!(
                "llama-completion returned non-zero status: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let raw = String::from_utf8_lossy(&output.stdout);
        Ok(clean_generation(&raw, system_prompt, user_prompt))
    }
}

pub struct BedrockClient {
    pub model: String,
}

impl LlmClient for BedrockClient {
    fn answer(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        // Use AWS CLI Bedrock runtime so we avoid extra SDK plumbing in MVP.
        let prompt = format!("System:\n{}\n\nUser:\n{}", system_prompt, user_prompt);
        let messages = serde_json::json!([
            {
                "role": "user",
                "content": [
                    { "text": prompt }
                ]
            }
        ]);
        let inference = serde_json::json!({
            "maxTokens": 700,
            "temperature": 0.2
        });

        let mut cmd = Command::new("aws");
        cmd.arg("bedrock-runtime")
            .arg("converse")
            .arg("--model-id")
            .arg(&self.model)
            .arg("--messages")
            .arg(messages.to_string())
            .arg("--inference-config")
            .arg(inference.to_string())
            .arg("--output")
            .arg("json");

        if let Ok(region) = std::env::var("AWS_REGION") {
            if !region.trim().is_empty() {
                cmd.arg("--region").arg(region);
            }
        }

        let output = cmd.output().context(
            "failed to invoke AWS CLI for Bedrock; install aws cli and configure credentials",
        )?;

        if !output.status.success() {
            return Err(anyhow!(
                "bedrock converse failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let value: serde_json::Value = serde_json::from_slice(&output.stdout)
            .context("failed to parse bedrock JSON response")?;
        let text = value
            .pointer("/output/message/content/0/text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("bedrock response missing output.message.content[0].text"))?;
        Ok(text.trim().to_string())
    }
}

fn run_llama(bin: &str, model: &str, prompt: &str) -> Result<std::process::Output> {
    let mut cmd = Command::new(bin);
    cmd.arg("-m").arg(model);
    cmd.arg("-ngl").arg(llama_gpu_layers());
    if let Some(device) = llama_device() {
        cmd.arg("--device").arg(device);
    }
    cmd.arg("-c").arg(llama_ctx_size());
    if bin == "llama-cli" {
        cmd.arg("-st");
    } else if bin == "llama-completion" {
        cmd.arg("-no-cnv");
    }
    if llama_no_warmup() {
        cmd.arg("--no-warmup");
    }
    cmd.arg("-p")
        .arg(prompt)
        .arg("-n")
        .arg(llama_max_tokens())
        .arg("--temp")
        .arg("0.2");
    cmd.output()
        .with_context(|| format!("failed to invoke {}", bin))
}

fn llama_gpu_layers() -> String {
    if let Ok(raw) = std::env::var("FACTORLENS_LLAMA_GPU_LAYERS") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "0".to_string()
}

fn llama_device() -> Option<String> {
    if let Ok(raw) = std::env::var("FACTORLENS_LLAMA_DEVICE") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    None
}

fn llama_ctx_size() -> String {
    if let Ok(raw) = std::env::var("FACTORLENS_LLAMA_CTX") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "4096".to_string()
}

fn llama_max_tokens() -> String {
    if let Ok(raw) = std::env::var("FACTORLENS_LLAMA_MAX_TOKENS") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "96".to_string()
}

fn llama_no_warmup() -> bool {
    if let Ok(raw) = std::env::var("FACTORLENS_LLAMA_NO_WARMUP") {
        let lower = raw.trim().to_ascii_lowercase();
        return !(lower == "0" || lower == "false" || lower == "no");
    }
    true
}

fn clean_generation(raw: &str, system_prompt: &str, user_prompt: &str) -> String {
    let mut text = raw.replace("\r\n", "\n");
    if let Some(i) = text.rfind("\nassistant") {
        text = text[(i + "\nassistant".len())..].to_string();
    }
    if let Some(i) = text.find("\n> EOF by user") {
        text.truncate(i);
    }

    let cleaned = text
        .lines()
        .filter(|line| {
            let t = line.trim();
            !(t.is_empty()
                || t == "user"
                || t == "assistant"
                || t == ">"
                || t == "[SYSTEM]"
                || t == "[USER]"
                || t.starts_with("<|im_start|>")
                || t.starts_with("<|im_end|>")
                || t.starts_with("Question:")
                || t.starts_with("Artifact context:")
                || t.starts_with("k=")
                || t.starts_with("explained_variance=")
                || t.starts_with("outliers=")
                || t.starts_with("artifacts_dir="))
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();

    let mut out = if cleaned.is_empty() {
        raw.trim().to_string()
    } else {
        cleaned
    };

    let combined_prompt = format!("[SYSTEM]\n{}\n\n[USER]\n{}", system_prompt, user_prompt);
    if let Some(i) = out.find(&combined_prompt) {
        out = out[(i + combined_prompt.len())..].trim_start().to_string();
    }

    let mut out_trim = out.trim_start().to_string();
    if let Some(first_line_end) = out_trim.find('\n') {
        let first_line = out_trim[..first_line_end].trim();
        if first_line == system_prompt {
            out_trim = out_trim[(first_line_end + 1)..].trim_start().to_string();
        }
    } else if out_trim.trim() == system_prompt {
        out_trim.clear();
    }

    let mut out_trim = out_trim.trim_start().to_string();
    for prefix in [system_prompt, user_prompt] {
        if let Some(stripped) = out_trim.strip_prefix(prefix) {
            out_trim = stripped.trim_start().to_string();
        }
    }
    out = out_trim;

    if out.ends_with("assistant") {
        out = out.trim_end_matches("assistant").trim_end().to_string();
    }

    out
}
