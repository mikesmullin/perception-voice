//! YAML configuration, field-compatible with the Python `config.yml`.
//!
//! Fields the Rust build no longer has a use for (`transcription.model`,
//! `device`, `compute_type`, `beam_size`, the whole `realtime` and `muse`
//! blocks) are accepted and ignored rather than rejected, so one config file
//! can sit in front of either implementation during the swap.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

fn default_socket_path() -> String {
    "perception.sock".into()
}
fn default_retention() -> u64 {
    30
}
fn default_sample_rate() -> usize {
    16_000
}
fn default_buffer_size() -> usize {
    512
}
fn default_min_utterance() -> f32 {
    1.1
}
fn default_post_silence() -> f32 {
    0.6
}
fn default_pre_buffer() -> f32 {
    1.0
}
fn default_webrtc_sensitivity() -> u8 {
    3
}
fn default_silero_sensitivity() -> f32 {
    0.05
}
fn default_max_new_tokens() -> usize {
    128
}
fn default_threads() -> usize {
    0
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_socket_path")]
    pub socket_path: String,
    #[serde(default = "default_retention")]
    pub buffer_retention_minutes: u64,
    #[serde(default)]
    pub discard_phrases: Vec<String>,
    #[serde(default)]
    pub transcript_log_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: usize,
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,
    /// `None` selects the system default input, re-resolved on every open.
    #[serde(default)]
    pub mic_device: Option<String>,
    #[serde(default = "default_min_utterance")]
    pub min_utterance_duration: f32,
    #[serde(default = "default_post_silence")]
    pub post_speech_silence_duration: f32,
    #[serde(default = "default_pre_buffer")]
    pub pre_recording_buffer_duration: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VadConfig {
    #[serde(default = "default_webrtc_sensitivity")]
    pub webrtc_sensitivity: u8,
    #[serde(default = "default_silero_sensitivity")]
    pub silero_sensitivity: f32,
    /// Path to `silero_vad.onnx`; defaults beside the ASR bundle.
    #[serde(default)]
    pub silero_model_path: Option<String>,
    /// `false` disables Silero and leaves WebRTC as the only gate.
    #[serde(default = "default_true")]
    pub silero_enabled: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Audio8Config {
    /// Bundle directory; defaults to `$AUDIO8_ASR_BUNDLE` or the cache path.
    #[serde(default)]
    pub bundle_dir: Option<String>,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    /// ONNX Runtime intra-op threads. 0 lets the runtime decide.
    #[serde(default = "default_threads")]
    pub threads: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            socket_path: default_socket_path(),
            buffer_retention_minutes: default_retention(),
            discard_phrases: Vec::new(),
            transcript_log_path: None,
        }
    }
}
impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            buffer_size: default_buffer_size(),
            mic_device: None,
            min_utterance_duration: default_min_utterance(),
            post_speech_silence_duration: default_post_silence(),
            pre_recording_buffer_duration: default_pre_buffer(),
        }
    }
}
impl Default for VadConfig {
    fn default() -> Self {
        Self {
            webrtc_sensitivity: default_webrtc_sensitivity(),
            silero_sensitivity: default_silero_sensitivity(),
            silero_model_path: None,
            silero_enabled: true,
        }
    }
}
impl Default for Audio8Config {
    fn default() -> Self {
        Self {
            bundle_dir: None,
            max_new_tokens: default_max_new_tokens(),
            threads: default_threads(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub audio: AudioConfig,
    #[serde(default)]
    pub vad: VadConfig,
    #[serde(default)]
    pub audio8: Audio8Config,

    /// Directory the config was loaded from; relative paths anchor here.
    #[serde(skip)]
    pub base_dir: PathBuf,
}

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let path = match path {
            Some(p) => p.to_path_buf(),
            None => PathBuf::from("config.yml"),
        };
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let mut config: Config = serde_yaml::from_str(&text)
            .with_context(|| format!("parsing config {}", path.display()))?;
        config.base_dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(config)
    }

    /// Resolve a configured path against the config's directory.
    pub fn resolve(&self, value: &str) -> PathBuf {
        let p = Path::new(value);
        if p.is_absolute() {
            p.to_path_buf()
        } else if let Some(rest) = value.strip_prefix("~/") {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(rest)
        } else {
            self.base_dir.join(p)
        }
    }

    pub fn socket_path(&self) -> PathBuf {
        self.resolve(&self.server.socket_path)
    }

    pub fn transcript_log_path(&self) -> Option<PathBuf> {
        self.server
            .transcript_log_path
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| self.resolve(s))
    }

    pub fn bundle_dir(&self) -> PathBuf {
        match self.audio8.bundle_dir.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            Some(s) => self.resolve(s),
            None => audio8_asr::default_bundle_dir(),
        }
    }

    pub fn silero_model_path(&self) -> PathBuf {
        match self.vad.silero_model_path.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            Some(s) => self.resolve(s),
            None => self
                .bundle_dir()
                .parent()
                .unwrap_or(Path::new("."))
                .join("vad/silero_vad.onnx"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_fill_in_for_an_empty_document() {
        let c: Config = serde_yaml::from_str("{}").unwrap();
        assert_eq!(c.audio.sample_rate, 16_000);
        assert_eq!(c.audio.buffer_size, 512);
        assert_eq!(c.vad.webrtc_sensitivity, 3);
        assert_eq!(c.server.buffer_retention_minutes, 30);
    }

    #[test]
    fn unknown_python_only_fields_are_ignored() {
        // A config written for the Python build must still load here.
        let yaml = r#"
transcription:
  backend: "audio8"
  model: "large-v3-turbo"
  device: "cuda"
realtime:
  enabled: true
  model: "tiny.en"
muse:
  api_key: null
audio:
  sample_rate: 16000
  min_utterance_duration: 1.4
"#;
        let c: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(c.audio.min_utterance_duration, 1.4);
    }

    #[test]
    fn relative_paths_anchor_at_the_config_directory() {
        let mut c = Config::default();
        c.base_dir = PathBuf::from("/opt/pv");
        assert_eq!(c.resolve("perception.sock"), PathBuf::from("/opt/pv/perception.sock"));
        assert_eq!(c.resolve("/tmp/x.sock"), PathBuf::from("/tmp/x.sock"));
    }

    #[test]
    fn tilde_expands_to_home() {
        let mut c = Config::default();
        c.base_dir = PathBuf::from("/opt/pv");
        let home = std::env::var("HOME").unwrap();
        assert_eq!(c.resolve("~/x.log"), PathBuf::from(home).join("x.log"));
    }

    #[test]
    fn blank_transcript_log_disables_it() {
        let mut c = Config::default();
        c.server.transcript_log_path = Some("   ".into());
        assert!(c.transcript_log_path().is_none());
        c.server.transcript_log_path = Some("./tmp/convos.md".into());
        assert!(c.transcript_log_path().is_some());
    }
}
