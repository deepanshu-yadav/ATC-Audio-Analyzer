import os
import json
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from transcribe_atc import AudioTranscriptionPipeline
import database
import device_utils

# Base paths
BACKEND_DIR = Path(__file__).resolve().parent
STATIC_DIR = BACKEND_DIR / "static"
ASSETS_DIR = STATIC_DIR / "assets"
OUTPUT_DIR = BACKEND_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize SQLite database
database.init_db()

app = FastAPI(
    title="Air Traffic Control ASR API",
    description="ASR Transcription API for ATC audio files using CrispASR (CPU/CUDA, Windows/Linux)",
    version="1.0.0"
)

# Mount outputs for previewing generated audio files
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Mount assets directory for frontend script/style bundles
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# Mount static files folder under /static
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
@app.get("/index.html")
def read_root():
    """Serves the frontend single-page application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        status_code=404,
        content={"detail": "Frontend index.html not found. Please build frontend with 'npm run build' and copy to static/."}
    )


@app.get("/api/system")
def get_system_status():
    """
    Returns detected platform information, available compute binaries (CPU/CUDA),
    and speech model availability.
    """
    return device_utils.get_system_capabilities()


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    apply_noise_removal: bool = Form(True),
    noise_reduction: float = Form(0.75),
    language: str = Form("en"),
    save_intermediate: bool = Form(True),
    device: str = Form("auto")
):
    """
    Upload an audio file to transcribe it with the ATC pipeline.
    
    Parameters:
    - **file**: The audio file (WAV, MP3, etc.)
    - **apply_noise_removal**: Whether to apply bandpass filter + spectral noise reduction
    - **noise_reduction**: Denoising coefficient (0.0 to 1.0)
    - **language**: Target language code (default 'en')
    - **save_intermediate**: If true, intermediate filtered/denoised overlapping 30s chunks are saved to the 'output' directory.
    - **device**: Compute device to use ('auto', 'cuda', or 'cpu')
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
        
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        temp_input_path = tmp.name
        
    try:
        # Read the uploaded file
        content = await file.read()
        with open(temp_input_path, "wb") as f:
            f.write(content)
            
        # Discover binary and model paths based on device and OS
        try:
            exe_path, resolved_device, use_gpu = device_utils.find_crispasr_binary(device=device)
            model_path = device_utils.find_model_path()
        except FileNotFoundError as fnf_err:
            raise HTTPException(status_code=500, detail=str(fnf_err))
            
        req_pipeline = AudioTranscriptionPipeline(
            exe_path=exe_path,
            model_path=model_path,
            sample_rate=16000,
            use_gpu=use_gpu,
            device=resolved_device
        )
        
        # Run transcription pipeline
        result = req_pipeline.process_audio(
            audio_path=temp_input_path,
            apply_vad=True,
            apply_noise_removal=apply_noise_removal,
            noise_reduction=noise_reduction,
            language=language,
            save_intermediate=save_intermediate,
            output_dir=str(OUTPUT_DIR)
        )
        
        # Save transcription JSON to output folder matching standard CLI behavior
        import datetime
        timestamp = result.get("timestamp", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        original_name = Path(file.filename).stem
        json_output_path = os.path.join(str(OUTPUT_DIR), f"{original_name}_transcription_{timestamp}.json")
        req_pipeline.save_transcription(result, json_output_path)
        
        # Format the response data explicitly
        response_data = {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "language": result.get("language", ""),
            "processing_time_seconds": result.get("processing_time_seconds", 0.0),
            "speech_duration_seconds": result.get("speech_duration_seconds", 0.0),
            "vad_enabled": result.get("vad_enabled", True),
            "noise_removal_enabled": result.get("noise_removal_enabled", True),
            "device": resolved_device,
            "requested_device": device,
            "timestamp": timestamp,
            "saved_transcription_file": json_output_path,
            "filtered_full_path": result.get("filtered_full_path"),
            "filtered_full_path_rel": result.get("filtered_full_path_rel"),
            "denoised_full_path": result.get("denoised_full_path"),
            "denoised_full_path_rel": result.get("denoised_full_path_rel"),
            "intermediate_chunks": result.get("intermediate_chunks", [])
        }
        
        # Save transcription to database
        try:
            inserted_id = database.save_transcription(
                filename=file.filename,
                text=response_data["text"],
                segments=response_data["segments"],
                language=response_data["language"],
                processing_time_seconds=response_data["processing_time_seconds"],
                speech_duration_seconds=response_data["speech_duration_seconds"],
                vad_enabled=response_data["vad_enabled"],
                noise_removal_enabled=response_data["noise_removal_enabled"],
                device=response_data["device"],
                timestamp=response_data["timestamp"],
                saved_transcription_file=response_data["saved_transcription_file"],
                filtered_full_path_rel=response_data["filtered_full_path_rel"],
                denoised_full_path_rel=response_data["denoised_full_path_rel"],
                intermediate_chunks=response_data["intermediate_chunks"]
            )
            response_data["id"] = inserted_id
        except Exception as db_err:
            print(f"Warning: Failed to save transcription to database: {db_err}")
            
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        
    finally:
        # Clean up temporary input file
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except Exception as e:
                print(f"Warning: Could not remove temp file {temp_input_path}: {e}")


@app.get("/transcriptions")
def get_transcriptions(page: int = 1, limit: int = 10):
    """Retrieve paginated historical transcriptions."""
    if page < 1:
        page = 1
    if limit < 1:
        limit = 10
    transcriptions, total = database.get_transcriptions(page, limit)
    import math
    pages = math.ceil(total / limit) if total > 0 else 1
    return {
        "transcriptions": transcriptions,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": pages
    }


class SegmentItem(BaseModel):
    start: float
    end: float
    text: str


class UpdateTranscriptionRequest(BaseModel):
    segments: List[SegmentItem]
    text: Optional[str] = None


@app.get("/transcriptions/{id}")
def get_transcription(id: int):
    """Retrieve full details of a specific transcription record."""
    record = database.get_transcription_by_id(id)
    if not record:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return record


@app.put("/transcriptions/{id}")
def update_transcription_record(id: int, payload: UpdateTranscriptionRequest):
    """
    Updates the timestamps and text of transcription segments in SQLite database
    and synchronizes with saved JSON file on disk if available.
    """
    existing = database.get_transcription_by_id(id)
    if not existing:
        raise HTTPException(status_code=404, detail="Transcription not found")
        
    segments_dict = [
        {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()}
        for seg in payload.segments
    ]
    
    full_text = payload.text if payload.text else " ".join(s["text"] for s in segments_dict if s["text"]).strip()
    
    updated = database.update_transcription(id, segments_dict, full_text)
    
    # Synchronize with the JSON file on disk if present
    json_path = existing.get("saved_transcription_file")
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                disk_data = json.load(f)
            disk_data["segments"] = segments_dict
            disk_data["text"] = full_text
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(disk_data, f, indent=2, ensure_ascii=False)
        except Exception as f_err:
            print(f"Warning: Could not update disk JSON file {json_path}: {f_err}")
            
    return updated


class SaveTranscriptionPayload(BaseModel):
    id: Optional[int] = None
    filename: Optional[str] = "custom_transcription.wav"
    text: Optional[str] = None
    segments: List[SegmentItem]
    language: Optional[str] = "en"
    processing_time_seconds: Optional[float] = 0.0
    speech_duration_seconds: Optional[float] = 0.0
    vad_enabled: Optional[bool] = True
    noise_removal_enabled: Optional[bool] = True
    device: Optional[str] = "cpu"
    timestamp: Optional[str] = None
    saved_transcription_file: Optional[str] = None
    filtered_full_path_rel: Optional[str] = None
    denoised_full_path_rel: Optional[str] = None
    intermediate_chunks: Optional[List[Dict[str, Any]]] = []


@app.post("/transcriptions/save")
def save_or_update_transcription(payload: SaveTranscriptionPayload):
    """
    Saves a new transcription or updates an existing one in the SQLite database.
    """
    segments_dict = [
        {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()}
        for seg in payload.segments
    ]
    full_text = payload.text if payload.text else " ".join(s["text"] for s in segments_dict if s["text"]).strip()
    
    if payload.id:
        existing = database.get_transcription_by_id(payload.id)
        if existing:
            updated = database.update_transcription(payload.id, segments_dict, full_text)
            # Sync to disk if applicable
            json_path = existing.get("saved_transcription_file")
            if json_path and os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        disk_data = json.load(f)
                    disk_data["segments"] = segments_dict
                    disk_data["text"] = full_text
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(disk_data, f, indent=2, ensure_ascii=False)
                except Exception as f_err:
                    print(f"Warning: Could not update disk JSON file {json_path}: {f_err}")
            return updated

    # Otherwise insert as new record
    import datetime
    ts = payload.timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    speech_dur = payload.speech_duration_seconds
    if not speech_dur and segments_dict:
        speech_dur = round(max(s["end"] for s in segments_dict), 2)
        
    inserted_id = database.save_transcription(
        filename=payload.filename or "transcription.wav",
        text=full_text,
        segments=segments_dict,
        language=payload.language or "en",
        processing_time_seconds=payload.processing_time_seconds or 0.0,
        speech_duration_seconds=speech_dur or 0.0,
        vad_enabled=payload.vad_enabled if payload.vad_enabled is not None else True,
        noise_removal_enabled=payload.noise_removal_enabled if payload.noise_removal_enabled is not None else True,
        device=payload.device or "cpu",
        timestamp=ts,
        saved_transcription_file=payload.saved_transcription_file or "",
        filtered_full_path_rel=payload.filtered_full_path_rel or "",
        denoised_full_path_rel=payload.denoised_full_path_rel or "",
        intermediate_chunks=payload.intermediate_chunks or []
    )
    return database.get_transcription_by_id(inserted_id)


@app.delete("/transcriptions/{id}")
def delete_transcription_record(id: int):
    """Deletes a transcription record from the database."""
    deleted = database.delete_transcription(id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return {"status": "success", "message": f"Transcription {id} deleted successfully", "id": id}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
