//! Time-indexed utterance buffer with per-client read markers.
//!
//! Clients poll: `set` stamps a marker at "now", `get` returns everything
//! transcribed since that client's marker. A client that has never called
//! `set` gets nothing rather than the whole retained history — the Python
//! behaved the same way, and it matters, because the alternative is the
//! keyboard typing half an hour of backlog the first time it connects.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local};

/// Lowercase, alphanumerics only — the normalization used for discard matching.
pub fn normalize_phrase(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

#[derive(Debug, Clone)]
pub struct Utterance {
    pub timestamp: DateTime<Local>,
    pub text: String,
    /// P95 windowed-RMS level in dBFS, when measured.
    pub level_db: Option<f32>,
}

impl Utterance {
    /// One JSONL record, with the millisecond-precision ISO 8601 timestamp the
    /// clients parse.
    pub fn to_jsonl(&self) -> String {
        serde_json::json!({
            "ts": self.timestamp.to_rfc3339_opts(chrono::SecondsFormat::Millis, false),
            "text": self.text,
        })
        .to_string()
    }

    /// Human-readable transcript line, e.g.
    /// `Sun, Sep 6 @ 4:09p | P95 -23dB | hello world`.
    pub fn to_log_line(&self) -> String {
        let ts = self.timestamp;
        let hour24 = ts.format("%H").to_string().parse::<u32>().unwrap_or(0);
        let hour12 = if hour24 % 12 == 0 { 12 } else { hour24 % 12 };
        let ampm = if hour24 < 12 { "a" } else { "p" };
        let stamp = format!(
            "{} {} @ {hour12}:{}{ampm}",
            ts.format("%a, %b"),
            ts.format("%-d"),
            ts.format("%M")
        );
        match self.level_db {
            Some(db) => format!("{stamp} | P95 {db:.0}dB | {}", self.text),
            None => format!("{stamp} | {}", self.text),
        }
    }
}

pub struct TranscriptionBuffer {
    retention_minutes: u64,
    utterances: Vec<Utterance>,
    markers: HashMap<String, DateTime<Local>>,
    discard: Vec<String>,
    log_path: Option<PathBuf>,
}

impl TranscriptionBuffer {
    pub fn new(
        retention_minutes: u64,
        discard_phrases: &[String],
        log_path: Option<PathBuf>,
    ) -> Self {
        Self {
            retention_minutes,
            utterances: Vec::new(),
            markers: HashMap::new(),
            discard: discard_phrases.iter().map(|p| normalize_phrase(p)).collect(),
            log_path,
        }
    }

    fn should_discard(&self, text: &str) -> bool {
        if self.discard.is_empty() {
            return false;
        }
        let n = normalize_phrase(text);
        self.discard.iter().any(|d| *d == n)
    }

    /// Returns `true` if stored, `false` if blank or a discard phrase.
    pub fn add(
        &mut self,
        text: &str,
        timestamp: DateTime<Local>,
        level_db: Option<f32>,
    ) -> bool {
        let text = text.trim();
        if text.is_empty() || self.should_discard(text) {
            return false;
        }
        let utterance = Utterance {
            timestamp,
            text: text.to_string(),
            level_db,
        };
        if let Some(path) = &self.log_path {
            if let Err(e) = append_line(path, &utterance.to_log_line()) {
                tracing::warn!("could not append to transcript log: {e}");
            }
        }
        self.utterances.push(utterance);
        self.prune();
        true
    }

    fn prune(&mut self) {
        let cutoff = Local::now() - chrono::Duration::minutes(self.retention_minutes as i64);
        self.utterances.retain(|u| u.timestamp >= cutoff);
    }

    pub fn set_marker(&mut self, uid: &str) {
        self.markers.insert(uid.to_string(), Local::now());
    }

    /// JSONL of everything after this uid's marker, and advance the marker.
    pub fn get_since_marker(&mut self, uid: &str) -> String {
        let Some(&marker) = self.markers.get(uid) else {
            return String::new();
        };
        let lines: Vec<String> = self
            .utterances
            .iter()
            .filter(|u| u.timestamp > marker)
            .map(|u| u.to_jsonl())
            .collect();
        if let Some(last) = self.utterances.iter().filter(|u| u.timestamp > marker).last() {
            self.markers.insert(uid.to_string(), last.timestamp);
        }
        lines.join("\n")
    }

    pub fn len(&self) -> usize {
        self.utterances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.utterances.is_empty()
    }
}

fn append_line(path: &Path, line: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{line}")
}
