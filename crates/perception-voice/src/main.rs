//! perception-voice: system-level speech-to-text service.

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use perception_voice::{client, config::Config, server};

const USAGE: &str = "\
perception-voice - system-level speech-to-text service

USAGE:
    perception-voice [-c CONFIG] serve [-v]
    perception-voice [-c CONFIG] client set <uid>
    perception-voice [-c CONFIG] client get <uid>
    perception-voice [-c CONFIG] client stream <levels|words>

OPTIONS:
    -c, --config FILE   config file (default: config.yml in the working directory)
    -v, --verbose       debug-level logging
    -h, --help          this message
";

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    let mut config_path: Option<PathBuf> = None;
    let mut verbose = false;
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            "-v" | "--verbose" => verbose = true,
            "-c" | "--config" => {
                i += 1;
                config_path = Some(PathBuf::from(
                    args.get(i).ok_or_else(|| anyhow!("--config needs a path"))?,
                ));
            }
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    args.clear();

    if positional.is_empty() {
        print!("{USAGE}");
        return Err(anyhow!("no command given"));
    }

    init_logging(verbose);
    let config = Config::load(config_path.as_deref())?;

    match positional[0].as_str() {
        "serve" => server::serve(config),
        "client" => {
            let socket = config.socket_path();
            match positional.get(1).map(String::as_str) {
                Some("set") => {
                    let uid = positional.get(2).ok_or_else(|| anyhow!("set needs a uid"))?;
                    client::set_marker(&socket, uid)
                }
                Some("get") => {
                    let uid = positional.get(2).ok_or_else(|| anyhow!("get needs a uid"))?;
                    let text = client::get_transcriptions(&socket, uid)?;
                    if !text.is_empty() {
                        println!("{text}");
                    }
                    Ok(())
                }
                Some("stream") => {
                    let channel = positional
                        .get(2)
                        .ok_or_else(|| anyhow!("stream needs a channel: levels or words"))?;
                    client::stream(&socket, channel)
                }
                _ => {
                    print!("{USAGE}");
                    Err(anyhow!("unknown client subcommand"))
                }
            }
        }
        other => {
            print!("{USAGE}");
            Err(anyhow!("unknown command: {other}"))
        }
    }
}

fn init_logging(verbose: bool) {
    let default = if verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default)),
        )
        .with_target(false)
        .init();
}
