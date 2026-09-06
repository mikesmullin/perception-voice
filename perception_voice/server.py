"""
perception-voice server

Main server daemon that:
- Loads Whisper model
- Captures audio and transcribes continuously
- Stores utterances in memory buffer
- Handles client requests via Unix socket
"""

import logging
import os
import signal
import socket
import sys
import threading
from pathlib import Path
from typing import Optional

from perception_voice.buffer import TranscriptionBuffer
from perception_voice.config import Config
from perception_voice.ipc import (
    create_server_socket,
    make_error_response,
    make_ok_response,
    recv_message,
    send_message,
)
from perception_voice.streams import CHANNELS, StreamHub
from perception_voice.transcriber import Transcriber

logger = logging.getLogger(__name__)


class Server:
    """
    perception-voice server daemon
    
    Manages:
    - Whisper transcription
    - Transcription buffer
    - Client connections via Unix socket
    """
    
    def __init__(self, config: Config, verbose: bool = False):
        """
        Initialize server
        
        Args:
            config: Server configuration
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._transcriber: Optional[Transcriber] = None
        self._buffer: Optional[TranscriptionBuffer] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._hub: Optional[StreamHub] = None
    
    def run(self) -> None:
        """Run the server (blocking)"""
        self._running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Initialize buffer (+ optional append-only JSONL transcript log)
        log_path = self.config.get_transcript_log_path()
        self._buffer = TranscriptionBuffer(
            retention_minutes=self.config.server.buffer_retention_minutes,
            discard_phrases=self.config.server.discard_phrases,
            log_path=log_path,
        )
        logger.info(
            f"Buffer initialized (retention: {self.config.server.buffer_retention_minutes} min"
            f"{f', transcript log: {log_path}' if log_path else ''})"
        )
        
        # Initialize push-stream hub (levels + words subscriptions)
        self._hub = StreamHub(sample_rate=self.config.audio.sample_rate)

        # Initialize transcriber
        self._transcriber = Transcriber(
            audio_config=self.config.audio,
            transcription_config=self.config.transcription,
            vad_config=self.config.vad,
            on_transcription=self._on_transcription,
            verbose=self.verbose,
            stream_hub=self._hub,
            realtime_config=self.config.realtime,
            muse_config=self.config.muse,
            audio8_config=self.config.audio8,
        )
        
        # Start transcriber
        self._transcriber.start()
        
        # Setup socket
        socket_path = self.config.get_socket_path()
        self._server_socket = create_server_socket(socket_path)
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)  # Allow periodic shutdown check
        
        logger.info(f"Server listening on {socket_path}")
        
        # Accept connections in main thread
        self._accept_connections()
        
        # Cleanup
        self._cleanup()
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False
    
    def _on_transcription(self, text: str, timestamp=None, end_timestamp=None,
                            level_db=None) -> None:
        """Callback for new transcriptions"""
        if self._buffer:
            added = self._buffer.add(text, timestamp, level_db=level_db)
            if self.verbose:
                word_count = len(text.split())
                level = f" (P95 {level_db:.0f}dB)" if level_db is not None else ""
                if added:
                    logger.info(f"Buffered: {word_count} words{level}")
                else:
                    logger.info(f"Discarded: {word_count} words (matched discard phrase)")

        # Push to `subscribe words` subscribers (Ada brain) the moment the
        # utterance is finalized — no polling anywhere in Ada's pipeline
        if self._hub:
            import time as _time
            t_start = timestamp.timestamp() if timestamp else None
            t_end = end_timestamp.timestamp() if end_timestamp else None
            self._hub.publish_words({
                "ev": "utterance",
                "ts": _time.time(),
                "text": text,
                "t_start": t_start,
                "t_end": t_end,
            })
    
    def _accept_connections(self) -> None:
        """Accept and handle client connections"""
        while self._running:
            try:
                client_sock, _ = self._server_socket.accept()
                # Handle client in separate thread
                threading.Thread(
                    target=self._handle_client,
                    args=(client_sock,),
                    daemon=True,
                ).start()
            except socket.timeout:
                continue
            except OSError as e:
                if self._running:
                    logger.error(f"Accept error: {e}")
                break
    
    def _handle_client(self, client_sock: socket.socket) -> None:
        """Handle a single client connection"""
        handed_off = False
        try:
            request = recv_message(client_sock)
            if not request:
                return

            # `subscribe` turns the connection into a long-lived push stream
            # owned by the hub; everything else stays request/response.
            if request.get("command") == "subscribe":
                channel = request.get("channel")
                if channel not in CHANNELS:
                    send_message(client_sock, make_error_response(
                        f"unknown channel: {channel} (expected one of {CHANNELS})"
                    ))
                    return
                send_message(client_sock, {"status": "ok", "channel": channel})
                self._hub.add_subscriber(channel, client_sock)
                handed_off = True
                return

            response = self._process_request(request)
            send_message(client_sock, response)

        except Exception as e:
            logger.error(f"Client handler error: {e}")
            try:
                send_message(client_sock, make_error_response(str(e)))
            except Exception:
                pass
        finally:
            if not handed_off:
                try:
                    client_sock.close()
                except Exception:
                    pass
    
    def _process_request(self, request: dict) -> dict:
        """Process a client request and return response"""
        command = request.get("command")
        uid = request.get("uid")
        
        if not command:
            return make_error_response("missing 'command' field")
        
        if command == "set":
            if not uid:
                return make_error_response("missing 'uid' field")
            self._buffer.set_marker(uid)
            if self.verbose:
                logger.info(f"Set marker for '{uid}'")
            return make_ok_response()
        
        elif command == "get":
            if not uid:
                return make_error_response("missing 'uid' field")
            text = self._buffer.get_since_marker(uid)
            if self.verbose:
                line_count = len(text.split('\n')) if text else 0
                logger.info(f"Get for '{uid}': {line_count} utterances")
            return make_ok_response(text=text)
        
        else:
            return make_error_response(f"unknown command: {command}")
    
    def _cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("Cleaning up...")
        
        # Stop transcriber
        if self._transcriber:
            self._transcriber.stop()
        
        # Close socket
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
        
        # Remove socket file
        socket_path = self.config.get_socket_path()
        if socket_path.exists():
            try:
                socket_path.unlink()
            except Exception:
                pass
        
        logger.info("Server stopped")
        
        # Force exit to ensure all threads terminate
        # This is necessary because sounddevice and other libraries may have
        # background threads that don't respond to normal shutdown
        os._exit(0)


def run_server(config: Config, verbose: bool = False) -> None:
    """
    Run the perception-voice server
    
    Args:
        config: Server configuration
        verbose: Enable verbose logging
    """
    server = Server(config, verbose=verbose)
    server.run()
