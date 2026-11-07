"""
Time-indexed transcription buffer with configurable TTL

Stores utterances with timestamps and provides per-client read markers.
"""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def normalize_phrase(text: str) -> str:
    """Normalize text for comparison: lowercase, alphanumeric only"""
    return re.sub(r'[^a-z0-9]', '', text.lower().strip())


@dataclass
class Utterance:
    """A single transcribed utterance with timestamp"""
    timestamp: datetime
    text: str
    
    def to_jsonl(self) -> str:
        """Convert to JSONL format with ISO 8601 timestamp"""
        ts_str = self.timestamp.isoformat(timespec='milliseconds')
        return json.dumps({"ts": ts_str, "text": self.text})
    
    @classmethod
    def from_text(cls, text: str, timestamp: Optional[datetime] = None) -> "Utterance":
        """Create utterance with given or current timestamp"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).astimezone()
        return cls(timestamp=timestamp, text=text)


class TranscriptionBuffer:
    """
    Thread-safe buffer for storing transcriptions with timestamps
    
    Features:
    - Configurable retention period (TTL)
    - Per-client read markers
    - Automatic cleanup of old entries
    - Discard phrases filtering
    """
    
    def __init__(self, retention_minutes: int = 30, discard_phrases: List[str] = None):
        """
        Initialize buffer
        
        Args:
            retention_minutes: How long to keep utterances in memory
            discard_phrases: List of phrases to discard (normalized matching)
        """
        self.retention_minutes = retention_minutes
        self._utterances: List[Utterance] = []
        self._read_markers: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        
        # Pre-normalize discard phrases for fast matching
        self._discard_phrases: set = set()
        if discard_phrases:
            for phrase in discard_phrases:
                self._discard_phrases.add(normalize_phrase(phrase))
    
    def _should_discard(self, text: str) -> bool:
        """Check if text matches a discard phrase"""
        if not self._discard_phrases:
            return False
        normalized = normalize_phrase(text)
        return normalized in self._discard_phrases
    
    def add(self, text: str, timestamp: Optional[datetime] = None) -> bool:
        """
        Add a new transcribed utterance
        
        Args:
            text: The transcribed text
            timestamp: When the utterance started (default: now)
        
        Returns:
            True if added, False if discarded
        """
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        # Check discard phrases
        if self._should_discard(text):
            logger.debug(f"Discarded phrase: {text!r}")
            return False
        
        utterance = Utterance.from_text(text, timestamp)
        
        with self._lock:
            self._utterances.append(utterance)
            self._cleanup_old_entries()
        
        logger.debug(f"Added utterance: {len(text)} chars at {utterance.timestamp}")
        return True
    
    def set_marker(self, uid: str) -> None:
        """
        Set read marker for a client to current time
        
        Args:
            uid: Unique client identifier
        """
        now = datetime.now(timezone.utc).astimezone()
        with self._lock:
            self._read_markers[uid] = now
        logger.debug(f"Set marker for '{uid}' to {now}")
    
    def get_since_marker(self, uid: str) -> str:
        """
        Get all utterances since the client's read marker as JSONL
        
        Updates the read marker to the latest returned utterance's timestamp.
        
        Args:
            uid: Unique client identifier
        
        Returns:
            JSONL string of utterances, or empty string if none
        """
        with self._lock:
            marker = self._read_markers.get(uid)
            
            if marker is None:
                # No marker set, return empty and set marker to now
                self._read_markers[uid] = datetime.now(timezone.utc).astimezone()
                return ""
            
            # Find utterances since marker
            result_lines = []
            latest_timestamp = marker
            for utterance in self._utterances:
                if utterance.timestamp > marker:
                    result_lines.append(utterance.to_jsonl())
                    if utterance.timestamp > latest_timestamp:
                        latest_timestamp = utterance.timestamp
            
            # Update marker to latest returned utterance (not "now")
            # This prevents the marker from racing ahead of in-flight transcriptions
            if result_lines:
                self._read_markers[uid] = latest_timestamp
                logger.debug(f"Returning {len(result_lines)} utterances for '{uid}'")
        
        return "\n".join(result_lines)
    
    def _cleanup_old_entries(self) -> None:
        """Remove entries older than retention period (called with lock held)"""
        if not self._utterances:
            return
        
        cutoff = datetime.now(timezone.utc).astimezone()
        cutoff_seconds = self.retention_minutes * 60
        
        # Find first entry to keep
        keep_from = 0
        for i, utterance in enumerate(self._utterances):
            age_seconds = (cutoff - utterance.timestamp).total_seconds()
            if age_seconds <= cutoff_seconds:
                keep_from = i
                break
        else:
            # All entries are old
            keep_from = len(self._utterances)
        
        if keep_from > 0:
            removed = keep_from
            self._utterances = self._utterances[keep_from:]
            logger.debug(f"Cleaned up {removed} old entries")
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        with self._lock:
            return {
                "utterance_count": len(self._utterances),
                "marker_count": len(self._read_markers),
                "retention_minutes": self.retention_minutes,
            }
