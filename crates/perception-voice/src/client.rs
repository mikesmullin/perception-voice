//! Client side of the IPC protocol — the `perception-voice client ...`
//! subcommands, used for scripting and for debugging a running server.

use std::io::Read;
use std::path::Path;

use anyhow::{anyhow, Result};
use serde_json::json;

use crate::ipc;
use crate::streams::{FLAG_END, FLAG_MAGIC_CHECK, FLAG_START, FLAG_VAD, FRAME_SIZE};

pub fn set_marker(socket: &Path, uid: &str) -> Result<()> {
    let mut stream = ipc::connect(socket)?;
    ipc::send_json(&mut stream, &json!({"command": "set", "uid": uid}))?;
    expect_ok(&mut stream)?;
    Ok(())
}

pub fn get_transcriptions(socket: &Path, uid: &str) -> Result<String> {
    let mut stream = ipc::connect(socket)?;
    ipc::send_json(&mut stream, &json!({"command": "get", "uid": uid}))?;
    let response = expect_ok(&mut stream)?;
    Ok(response
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string())
}

/// Subscribe and print. `words` prints one JSON event per line; `levels`
/// decodes the binary frames and prints roughly four lines a second.
pub fn stream(socket: &Path, channel: &str) -> Result<()> {
    let mut stream = ipc::connect(socket)?;
    ipc::send_json(
        &mut stream,
        &json!({"command": "subscribe", "channel": channel}),
    )?;
    expect_ok(&mut stream)?;

    if channel == "words" {
        while let Some(bytes) = ipc::recv_bytes(&mut stream)? {
            println!("{}", String::from_utf8_lossy(&bytes));
        }
        return Ok(());
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut scratch = [0u8; 4096];
    let mut seen = 0u64;
    loop {
        let n = stream.read(&mut scratch)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&scratch[..n]);
        while buf.len() >= FRAME_SIZE {
            let frame: Vec<u8> = buf.drain(..FRAME_SIZE).collect();
            let magic = u32::from_le_bytes(frame[0..4].try_into().unwrap());
            if magic != FLAG_MAGIC_CHECK {
                // Lost framing; drop what we have and resync on the next read.
                buf.clear();
                break;
            }
            let rms = f32::from_le_bytes(frame[8..12].try_into().unwrap());
            let bands: Vec<f32> = (0..4)
                .map(|i| {
                    let o = 12 + i * 4;
                    f32::from_le_bytes(frame[o..o + 4].try_into().unwrap())
                })
                .collect();
            let flags = u32::from_le_bytes(frame[32..36].try_into().unwrap());

            seen += 1;
            if seen % 16 == 0 || flags & (FLAG_START | FLAG_END) != 0 {
                let marks = format!(
                    "{}{}{}",
                    if flags & FLAG_VAD != 0 { "V" } else { "." },
                    if flags & FLAG_START != 0 { "S" } else { "" },
                    if flags & FLAG_END != 0 { "E" } else { "" },
                );
                let bar = "#".repeat((rms * 40.0) as usize);
                println!(
                    "rms={rms:0.2} bands=[{:0.2} {:0.2} {:0.2} {:0.2}] {marks} {bar}",
                    bands[0], bands[1], bands[2], bands[3]
                );
            }
        }
    }
    Ok(())
}

fn expect_ok(stream: &mut std::os::unix::net::UnixStream) -> Result<serde_json::Value> {
    let bytes = ipc::recv_bytes(stream)?.ok_or_else(|| anyhow!("no response from server"))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    if value.get("status").and_then(|v| v.as_str()) != Some("ok") {
        return Err(anyhow!(
            "server error: {}",
            value
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
        ));
    }
    Ok(value)
}
