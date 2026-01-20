"""
perception-voice client

CLI client for communicating with the perception-voice server.
"""

import logging
import sys
from pathlib import Path

from perception_voice.config import Config
from perception_voice.ipc import (
    create_client_socket,
    make_get_request,
    make_set_request,
    recv_message,
    send_message,
)

logger = logging.getLogger(__name__)

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def client_set(config: Config, uid: str) -> int:
    """
    Set read marker for a uid
    
    Args:
        config: Configuration
        uid: Unique client identifier
    
    Returns:
        Exit code
    """
    socket_path = config.get_socket_path()
    
    try:
        sock = create_client_socket(socket_path)
    except ConnectionError as e:
        logger.error(f"Cannot connect to server: {e}")
        return EXIT_ERROR
    
    try:
        send_message(sock, make_set_request(uid))
        response = recv_message(sock)
        
        if not response:
            logger.error("No response from server")
            return EXIT_ERROR
        
        if response.get("status") != "ok":
            logger.error(f"Server error: {response.get('message', 'unknown')}")
            return EXIT_ERROR
        
        return EXIT_SUCCESS
        
    except Exception as e:
        logger.error(f"Communication error: {e}")
        return EXIT_ERROR
    finally:
        try:
            sock.close()
        except Exception:
            pass


def client_get(config: Config, uid: str) -> int:
    """
    Get transcriptions since read marker for a uid
    
    Args:
        config: Configuration
        uid: Unique client identifier
    
    Returns:
        Exit code
    """
    socket_path = config.get_socket_path()
    
    try:
        sock = create_client_socket(socket_path)
    except ConnectionError as e:
        logger.error(f"Cannot connect to server: {e}")
        return EXIT_ERROR
    
    try:
        send_message(sock, make_get_request(uid))
        response = recv_message(sock)
        
        if not response:
            logger.error("No response from server")
            return EXIT_ERROR
        
        if response.get("status") != "ok":
            logger.error(f"Server error: {response.get('message', 'unknown')}")
            return EXIT_ERROR
        
        # Print text to stdout (empty string if no new text)
        text = response.get("text", "")
        if text:
            print(text)
        
        return EXIT_SUCCESS
        
    except Exception as e:
        logger.error(f"Communication error: {e}")
        return EXIT_ERROR
    finally:
        try:
            sock.close()
        except Exception:
            pass
