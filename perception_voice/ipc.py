"""
Unix domain socket IPC protocol

Provides JSON-based request/response communication between
perception-voice server and clients.
"""

import json
import logging
import os
import socket
import struct
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Message framing: 4-byte length prefix (big-endian) + JSON payload
HEADER_SIZE = 4
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB max message


def create_server_socket(socket_path: Path) -> socket.socket:
    """
    Create and bind a Unix domain socket for the server
    
    Args:
        socket_path: Path to the socket file
    
    Returns:
        Bound socket ready for listening
    """
    # Remove existing socket file
    if socket_path.exists():
        socket_path.unlink()
    
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(socket_path))
    
    # Set socket permissions (owner read/write only)
    os.chmod(socket_path, 0o600)
    
    return sock


def create_client_socket(socket_path: Path) -> socket.socket:
    """
    Create and connect a Unix domain socket for the client
    
    Args:
        socket_path: Path to the socket file
    
    Returns:
        Connected socket
    
    Raises:
        ConnectionError: If server is not running
    """
    if not socket_path.exists():
        raise ConnectionError(f"Server socket not found: {socket_path}")
    
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(socket_path))
    
    return sock


def send_message(sock: socket.socket, message: Dict[str, Any]) -> None:
    """
    Send a JSON message over the socket
    
    Args:
        sock: Connected socket
        message: Dictionary to send as JSON
    """
    payload = json.dumps(message).encode('utf-8')
    
    if len(payload) > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {len(payload)} bytes")
    
    # Send length prefix + payload
    header = struct.pack('>I', len(payload))
    sock.sendall(header + payload)


def recv_message(sock: socket.socket) -> Optional[Dict[str, Any]]:
    """
    Receive a JSON message from the socket
    
    Args:
        sock: Connected socket
    
    Returns:
        Parsed message dictionary, or None if connection closed
    """
    # Receive length prefix
    header = _recv_exact(sock, HEADER_SIZE)
    if not header:
        return None
    
    length = struct.unpack('>I', header)[0]
    
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {length} bytes")
    
    # Receive payload
    payload = _recv_exact(sock, length)
    if not payload:
        return None
    
    return json.loads(payload.decode('utf-8'))


def _recv_exact(sock: socket.socket, size: int) -> Optional[bytes]:
    """
    Receive exactly size bytes from socket
    
    Args:
        sock: Connected socket
        size: Number of bytes to receive
    
    Returns:
        Received bytes, or None if connection closed
    """
    data = b''
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data += chunk
    return data


# Request/Response helpers

def make_set_request(uid: str) -> Dict[str, Any]:
    """Create a 'set' command request"""
    return {"command": "set", "uid": uid}


def make_get_request(uid: str) -> Dict[str, Any]:
    """Create a 'get' command request"""
    return {"command": "get", "uid": uid}


def make_ok_response(text: Optional[str] = None) -> Dict[str, Any]:
    """Create a success response"""
    response: Dict[str, Any] = {"status": "ok"}
    if text is not None:
        response["text"] = text
    return response


def make_error_response(message: str) -> Dict[str, Any]:
    """Create an error response"""
    return {"status": "error", "message": message}
