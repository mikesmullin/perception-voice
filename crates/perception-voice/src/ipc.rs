//! Wire protocol: a 4-byte big-endian length prefix followed by JSON.
//!
//! Deliberately byte-identical to the Python implementation on `main`, so the
//! Rust server can serve the existing Python keyboard (and Ada) unchanged.
//! That is what makes an A/B swap safe: one `systemctl` change, no client
//! coordination, and reverting is the same change backwards.
//!
//! `subscribe` is the one command that does not follow request/response — it
//! converts the connection into a server-push stream owned by the stream hub.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

pub const HEADER_SIZE: usize = 4;
pub const MAX_MESSAGE_SIZE: usize = 1024 * 1024;

#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    pub command: String,
    #[serde(default)]
    pub uid: Option<String>,
    #[serde(default)]
    pub channel: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
}

impl Response {
    pub fn ok() -> Self {
        Self { status: "ok", message: None, text: None, channel: None }
    }
    pub fn ok_text(text: String) -> Self {
        Self { status: "ok", message: None, text: Some(text), channel: None }
    }
    pub fn ok_channel(channel: String) -> Self {
        Self { status: "ok", message: None, text: None, channel: Some(channel) }
    }
    pub fn error(message: impl Into<String>) -> Self {
        Self { status: "error", message: Some(message.into()), text: None, channel: None }
    }
}

pub fn send_bytes(stream: &mut impl Write, payload: &[u8]) -> Result<()> {
    if payload.len() > MAX_MESSAGE_SIZE {
        return Err(anyhow!("message too large: {} bytes", payload.len()));
    }
    stream.write_all(&(payload.len() as u32).to_be_bytes())?;
    stream.write_all(payload)?;
    stream.flush()?;
    Ok(())
}

pub fn send_json<T: Serialize>(stream: &mut impl Write, value: &T) -> Result<()> {
    send_bytes(stream, &serde_json::to_vec(value)?)
}

/// Read one framed message. `Ok(None)` means the peer closed cleanly.
pub fn recv_bytes(stream: &mut impl Read) -> Result<Option<Vec<u8>>> {
    let mut header = [0u8; HEADER_SIZE];
    match read_exact_or_eof(stream, &mut header)? {
        false => return Ok(None),
        true => {}
    }
    let len = u32::from_be_bytes(header) as usize;
    if len > MAX_MESSAGE_SIZE {
        return Err(anyhow!("declared message size too large: {len} bytes"));
    }
    let mut body = vec![0u8; len];
    if !read_exact_or_eof(stream, &mut body)? {
        return Ok(None);
    }
    Ok(Some(body))
}

pub fn recv_json<T: for<'de> Deserialize<'de>>(stream: &mut impl Read) -> Result<Option<T>> {
    match recv_bytes(stream)? {
        None => Ok(None),
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
    }
}

/// Like `read_exact`, but reports a clean EOF as `false` instead of an error.
fn read_exact_or_eof(stream: &mut impl Read, buf: &mut [u8]) -> Result<bool> {
    let mut filled = 0usize;
    while filled < buf.len() {
        match stream.read(&mut buf[filled..]) {
            Ok(0) => return Ok(false),
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(true)
}

pub fn connect(socket_path: &std::path::Path) -> Result<UnixStream> {
    if !socket_path.exists() {
        return Err(anyhow!("server socket not found: {}", socket_path.display()));
    }
    Ok(UnixStream::connect(socket_path)?)
}
