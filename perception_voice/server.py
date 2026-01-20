"""
perception-voice server

Main server daemon that:
- Loads Whisper model
- Captures audio and transcribes continuously
- Stores utterances in memory buffer
- Handles client requests via Unix socket
"""

import logging
import signal
import socket
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
    
    def run(self) -> None:
        """Run the server (blocking)"""
        self._running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Initialize buffer
        self._buffer = TranscriptionBuffer(
            retention_minutes=self.config.server.buffer_retention_minutes,
            discard_phrases=self.config.server.discard_phrases,
        )
        logger.info(
            f"Buffer initialized (retention: {self.config.server.buffer_retention_minutes} min)"
        )
        
        # Initialize transcriber
        self._transcriber = Transcriber(
            audio_config=self.config.audio,
            transcription_config=self.config.transcription,
            vad_config=self.config.vad,
            on_transcription=self._on_transcription,
            verbose=self.verbose,
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
    
    def _on_transcription(self, text: str) -> None:
        """Callback for new transcriptions"""
        if self._buffer:
            added = self._buffer.add(text)
            if self.verbose:
                word_count = len(text.split())
                if added:
                    logger.info(f"Buffered: {word_count} words")
                else:
                    logger.info(f"Discarded: {word_count} words (matched discard phrase)")
    
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
        try:
            request = recv_message(client_sock)
            if not request:
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


def run_server(config: Config, verbose: bool = False) -> None:
    """
    Run the perception-voice server
    
    Args:
        config: Server configuration
        verbose: Enable verbose logging
    """
    server = Server(config, verbose=verbose)
    server.run()
