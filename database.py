import sqlite3
import json
import os
from typing import List, Dict, Any, Tuple, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcriptions.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            text TEXT,
            segments TEXT, -- JSON string
            language TEXT,
            processing_time_seconds REAL,
            speech_duration_seconds REAL,
            vad_enabled INTEGER, -- 0 or 1
            noise_removal_enabled INTEGER, -- 0 or 1
            device TEXT,
            timestamp TEXT,
            saved_transcription_file TEXT,
            filtered_full_path_rel TEXT,
            denoised_full_path_rel TEXT,
            intermediate_chunks TEXT, -- JSON string
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_transcription(
    filename: str,
    text: str,
    segments: List[Dict[str, Any]],
    language: str,
    processing_time_seconds: float,
    speech_duration_seconds: float,
    vad_enabled: bool,
    noise_removal_enabled: bool,
    device: str,
    timestamp: str,
    saved_transcription_file: str,
    filtered_full_path_rel: str,
    denoised_full_path_rel: str,
    intermediate_chunks: List[Dict[str, Any]]
) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transcriptions (
            filename, text, segments, language, processing_time_seconds,
            speech_duration_seconds, vad_enabled, noise_removal_enabled,
            device, timestamp, saved_transcription_file,
            filtered_full_path_rel, denoised_full_path_rel, intermediate_chunks
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        text,
        json.dumps(segments),
        language,
        processing_time_seconds,
        speech_duration_seconds,
        1 if vad_enabled else 0,
        1 if noise_removal_enabled else 0,
        device,
        timestamp,
        saved_transcription_file,
        filtered_full_path_rel,
        denoised_full_path_rel,
        json.dumps(intermediate_chunks)
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_transcriptions(page: int = 1, limit: int = 10) -> Tuple[List[Dict[str, Any]], int]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM transcriptions")
    total = cursor.fetchone()[0]
    
    # Get paginated results ordered by newest first
    offset = (page - 1) * limit
    cursor.execute("""
        SELECT id, filename, text, language, processing_time_seconds,
               speech_duration_seconds, vad_enabled, noise_removal_enabled,
               device, timestamp, filtered_full_path_rel, denoised_full_path_rel, created_at
        FROM transcriptions
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))
    
    rows = cursor.fetchall()
    transcriptions = []
    for row in rows:
        transcriptions.append({
            "id": row["id"],
            "filename": row["filename"],
            "text": row["text"],
            "language": row["language"],
            "processing_time_seconds": row["processing_time_seconds"],
            "speech_duration_seconds": row["speech_duration_seconds"],
            "vad_enabled": bool(row["vad_enabled"]),
            "noise_removal_enabled": bool(row["noise_removal_enabled"]),
            "device": row["device"],
            "timestamp": row["timestamp"],
            "filtered_full_path_rel": row["filtered_full_path_rel"],
            "denoised_full_path_rel": row["denoised_full_path_rel"],
            "created_at": row["created_at"]
        })
    conn.close()
    return transcriptions, total

def get_transcription_by_id(transcription_id: int) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transcriptions WHERE id = ?", (transcription_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    return {
        "id": row["id"],
        "filename": row["filename"],
        "text": row["text"],
        "segments": json.loads(row["segments"]) if row["segments"] else [],
        "language": row["language"],
        "processing_time_seconds": row["processing_time_seconds"],
        "speech_duration_seconds": row["speech_duration_seconds"],
        "vad_enabled": bool(row["vad_enabled"]),
        "noise_removal_enabled": bool(row["noise_removal_enabled"]),
        "device": row["device"],
        "timestamp": row["timestamp"],
        "saved_transcription_file": row["saved_transcription_file"],
        "filtered_full_path_rel": row["filtered_full_path_rel"],
        "denoised_full_path_rel": row["denoised_full_path_rel"],
        "intermediate_chunks": json.loads(row["intermediate_chunks"]) if row["intermediate_chunks"] else [],
        "created_at": row["created_at"]
    }


def update_transcription(
    transcription_id: int,
    segments: List[Dict[str, Any]],
    text: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Updates transcription segments and text for a given transcription record.
    
    Args:
        transcription_id (int): Record ID to update.
        segments (List[Dict[str, Any]]): Updated list of segments.
        text (Optional[str]): Recomputed or provided full text.
        
    Returns:
        Optional[Dict[str, Any]]: Updated transcription record if found, else None.
    """
    if text is None:
        text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text")).strip()
        
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE transcriptions
        SET segments = ?, text = ?
        WHERE id = ?
    """, (json.dumps(segments), text, transcription_id))
    conn.commit()
    conn.close()
    
    return get_transcription_by_id(transcription_id)


def delete_transcription(transcription_id: int) -> bool:
    """
    Deletes a transcription record by ID.
    
    Args:
        transcription_id (int): Record ID to delete.
        
    Returns:
        bool: True if deleted, False otherwise.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transcriptions WHERE id = ?", (transcription_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted
