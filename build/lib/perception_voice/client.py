"""
perception-voice client

CLI client for communicating with the perception-voice server.
"""

import json
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
from perception_voice.streams import FLAG_END, FLAG_START, FLAG_VAD, FRAME_STRUCT

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


def client_stream(config: Config, channel: str) -> int:
    """
    Subscribe to a push stream and print it (debug tool for the Ada
    streaming interface; see /workspace/ada/docs/PROTOCOL.md).

    - words:  prints each JSON event as a line
    - levels: decodes binary FeatureFrames; prints ~4/s plus start/end
    """
    socket_path = config.get_socket_path()

    try:
        sock = create_client_socket(socket_path)
    except ConnectionError as e:
        logger.error(f"Cannot connect to server: {e}")
        return EXIT_ERROR

    try:
        send_message(sock, {"command": "subscribe", "channel": channel})
        ack = recv_message(sock)
        if not ack or ack.get("status") != "ok":
            logger.error(f"Subscribe failed: {ack}")
            return EXIT_ERROR

        if channel == "words":
            while True:
                msg = recv_message(sock)
                if msg is None:
                    break
                print(json.dumps(msg), flush=True)
        else:
            buf = b""
            n = 0
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                buf += data
                while len(buf) >= FRAME_STRUCT.size:
                    frame = buf[:FRAME_STRUCT.size]
                    buf = buf[FRAME_STRUCT.size:]
                    (_, _, rms, b0, b1, b2, b3, _, flags) = FRAME_STRUCT.unpack(frame)
                    n += 1
                    if n % 16 == 0 or flags & (FLAG_START | FLAG_END):
                        marks = "".join([
                            "V" if flags & FLAG_VAD else ".",
                            "S" if flags & FLAG_START else "",
                            "E" if flags & FLAG_END else "",
                        ])
                        bar = "#" * int(rms * 40)
                        print(f"rms={rms:0.2f} bands=[{b0:0.2f} {b1:0.2f} "
                              f"{b2:0.2f} {b3:0.2f}] {marks} {bar}", flush=True)
        return EXIT_SUCCESS

    except KeyboardInterrupt:
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
