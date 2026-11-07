//! Unix-socket server: request/response for `set`/`get`, and `subscribe` to
//! hand a connection over to the stream hub.

use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};

use crate::buffer::TranscriptionBuffer;
use crate::config::Config;
use crate::ipc::{self, Request, Response};
use crate::streams::{StreamHub, CHANNELS};
use crate::transcriber::Transcriber;

/// How long `accept` blocks before re-checking the shutdown flag.
///
/// The Python used 1.0s, which put up to a full second of dead time into
/// every restart (0.82s of a 2.98s restart-to-ready, measured). 100ms costs
/// 10 idle wakeups a second, which is nothing next to the audio callback.
const ACCEPT_TIMEOUT: Duration = Duration::from_millis(100);

pub fn serve(config: Config) -> Result<()> {
    let socket_path = config.socket_path();
    let buffer = Arc::new(Mutex::new(TranscriptionBuffer::new(
        config.server.buffer_retention_minutes,
        &config.server.discard_phrases,
        config.transcript_log_path(),
    )));
    tracing::info!(
        "buffer initialized (retention: {} min{})",
        config.server.buffer_retention_minutes,
        config
            .transcript_log_path()
            .map(|p| format!(", transcript log: {}", p.display()))
            .unwrap_or_default()
    );

    let hub = StreamHub::new(config.audio.sample_rate);

    // Capture and the socket come up now; the model loads behind them.
    let transcriber = Transcriber::start(&config, hub.clone(), buffer.clone())?;

    let listener = bind(&socket_path)?;
    listener.set_nonblocking(false)?;
    tracing::info!("server listening on {}", socket_path.display());

    let stop = Arc::new(AtomicBool::new(false));
    install_signal_handler(stop.clone())?;

    accept_loop(&listener, &stop, &buffer, &hub);

    tracing::info!("cleaning up...");
    transcriber.stop();
    drop(listener);
    let _ = std::fs::remove_file(&socket_path);
    tracing::info!("server stopped");
    Ok(())
}

fn bind(socket_path: &Path) -> Result<UnixListener> {
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    // A leftover socket file from a killed process would make bind fail; it is
    // safe to remove because a live server holds the inode, not the name.
    if socket_path.exists() {
        std::fs::remove_file(socket_path)
            .with_context(|| format!("removing stale socket {}", socket_path.display()))?;
    }
    UnixListener::bind(socket_path)
        .with_context(|| format!("binding {}", socket_path.display()))
}

fn accept_loop(
    listener: &UnixListener,
    stop: &Arc<AtomicBool>,
    buffer: &Arc<Mutex<TranscriptionBuffer>>,
    hub: &StreamHub,
) {
    // A short read timeout on the listener is not available directly, so poll
    // with nonblocking accept plus a sleep. This keeps shutdown prompt.
    listener.set_nonblocking(true).ok();
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let buffer = buffer.clone();
                let hub = hub.clone();
                std::thread::Builder::new()
                    .name("client".into())
                    .spawn(move || handle_client(stream, buffer, hub))
                    .ok();
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(ACCEPT_TIMEOUT);
            }
            Err(e) => {
                if !stop.load(Ordering::Relaxed) {
                    tracing::error!("accept error: {e}");
                }
                break;
            }
        }
    }
}

fn handle_client(
    mut stream: UnixStream,
    buffer: Arc<Mutex<TranscriptionBuffer>>,
    hub: StreamHub,
) {
    stream.set_nonblocking(false).ok();
    stream.set_read_timeout(Some(Duration::from_secs(10))).ok();

    let request: Request = match ipc::recv_json(&mut stream) {
        Ok(Some(r)) => r,
        Ok(None) => return,
        Err(e) => {
            tracing::warn!("malformed request: {e}");
            let _ = ipc::send_json(&mut stream, &Response::error(e.to_string()));
            return;
        }
    };

    match request.command.as_str() {
        "subscribe" => {
            let channel = request.channel.unwrap_or_default();
            if !CHANNELS.contains(&channel.as_str()) {
                let msg = format!("unknown channel: {channel} (expected one of {CHANNELS:?})");
                let _ = ipc::send_json(&mut stream, &Response::error(msg));
                return;
            }
            if ipc::send_json(&mut stream, &Response::ok_channel(channel.clone())).is_err() {
                return;
            }
            // Ownership moves to the hub; it closes the socket when the peer
            // goes away.
            hub.add_subscriber(&channel, stream);
        }
        "set" => {
            let response = match request.uid {
                Some(uid) if !uid.is_empty() => {
                    buffer.lock().expect("buffer").set_marker(&uid);
                    Response::ok()
                }
                _ => Response::error("missing 'uid' field"),
            };
            let _ = ipc::send_json(&mut stream, &response);
        }
        "get" => {
            let response = match request.uid {
                Some(uid) if !uid.is_empty() => {
                    let text = buffer.lock().expect("buffer").get_since_marker(&uid);
                    Response::ok_text(text)
                }
                _ => Response::error("missing 'uid' field"),
            };
            let _ = ipc::send_json(&mut stream, &response);
        }
        other => {
            let _ = ipc::send_json(
                &mut stream,
                &Response::error(format!("unknown command: {other}")),
            );
        }
    }
}

/// SIGTERM/SIGINT flip the flag; the accept loop notices within
/// `ACCEPT_TIMEOUT` and shutdown proceeds on the main thread.
///
/// `signal-hook` installs a handler that only touches an atomic, which is what
/// makes it async-signal-safe -- no allocation, no locks.
fn install_signal_handler(stop: Arc<AtomicBool>) -> Result<()> {
    for sig in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        signal_hook::flag::register(sig, stop.clone())
            .with_context(|| format!("registering handler for signal {sig}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn temp_socket(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("pv-test-{name}-{}.sock", std::process::id()));
        p
    }

    #[test]
    fn bind_removes_a_stale_socket_file() {
        let path = temp_socket("stale");
        std::fs::write(&path, b"not really a socket").unwrap();
        let listener = bind(&path).expect("bind over a stale file");
        drop(listener);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn set_then_get_round_trips_over_the_socket() {
        let path = temp_socket("roundtrip");
        let listener = bind(&path).unwrap();
        let buffer = Arc::new(Mutex::new(TranscriptionBuffer::new(30, &[], None)));
        let hub = StreamHub::new(16_000);

        let server_buffer = buffer.clone();
        let server_hub = hub.clone();
        let handle = std::thread::spawn(move || {
            // two connections: one `set`, one `get`
            for _ in 0..2 {
                let (stream, _) = listener.accept().unwrap();
                handle_client(stream, server_buffer.clone(), server_hub.clone());
            }
        });

        // set the marker
        let mut c = UnixStream::connect(&path).unwrap();
        ipc::send_bytes(&mut c, br#"{"command":"set","uid":"t"}"#).unwrap();
        let resp: serde_json::Value =
            serde_json::from_slice(&ipc::recv_bytes(&mut c).unwrap().unwrap()).unwrap();
        assert_eq!(resp["status"], "ok");
        drop(c);

        // an utterance arrives after the marker
        buffer
            .lock()
            .unwrap()
            .add("hello there", chrono::Local::now() + chrono::Duration::seconds(1), None);

        let mut c = UnixStream::connect(&path).unwrap();
        ipc::send_bytes(&mut c, br#"{"command":"get","uid":"t"}"#).unwrap();
        let resp: serde_json::Value =
            serde_json::from_slice(&ipc::recv_bytes(&mut c).unwrap().unwrap()).unwrap();
        assert_eq!(resp["status"], "ok");
        assert!(
            resp["text"].as_str().unwrap().contains("hello there"),
            "got {resp:?}"
        );

        handle.join().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn unknown_command_and_channel_are_rejected() {
        let path = temp_socket("reject");
        let listener = bind(&path).unwrap();
        let buffer = Arc::new(Mutex::new(TranscriptionBuffer::new(30, &[], None)));
        let hub = StreamHub::new(16_000);

        let b = buffer.clone();
        let h = hub.clone();
        let handle = std::thread::spawn(move || {
            for _ in 0..3 {
                let (stream, _) = listener.accept().unwrap();
                handle_client(stream, b.clone(), h.clone());
            }
        });

        for (payload, expect) in [
            (&br#"{"command":"nope"}"#[..], "unknown command"),
            (&br#"{"command":"subscribe","channel":"bogus"}"#[..], "unknown channel"),
            (&br#"{"command":"get"}"#[..], "missing 'uid'"),
        ] {
            let mut c = UnixStream::connect(&path).unwrap();
            c.write_all(&(payload.len() as u32).to_be_bytes()).unwrap();
            c.write_all(payload).unwrap();
            let resp: serde_json::Value =
                serde_json::from_slice(&ipc::recv_bytes(&mut c).unwrap().unwrap()).unwrap();
            assert_eq!(resp["status"], "error", "payload {payload:?}");
            assert!(
                resp["message"].as_str().unwrap().contains(expect),
                "expected {expect:?} in {resp:?}"
            );
        }

        handle.join().unwrap();
        std::fs::remove_file(&path).ok();
    }
}
