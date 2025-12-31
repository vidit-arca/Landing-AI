import os
from dotenv import load_dotenv

load_dotenv()
import io
import json
import uuid
import hashlib
import shutil as _shutil
import requests
import base64
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Configure LandingAI ADE
LANDING_API_KEY = os.getenv("LANDING_API_KEY")
ADE_PARSE_URL = "https://api.va.landing.ai/v1/ade/parse"
ADE_EXTRACT_URL = "https://api.va.landing.ai/v1/ade/extract"
HEADERS = {"Authorization": f"Basic {LANDING_API_KEY}"}

# Extraction schema
SCHEMA: Dict[str, Any] = {
  "type": "object",
  "title": "Extracted Markdown Document Data",
  "$schema": "http://json-schema.org/draft-07/schema#",
  "required": ["patient_info","report_dates","test_results","clinical_notes"],
  "properties": {
    "patient_info": {
      "type": "object",
      "required": ["name","age_gender","report_ref_id","patient_id","collected_datetime"],
      "properties": {
        "name": {"type": "string"},
        "age_gender": {"type": "string"},
        "patient_id": {"type": "string"},
        "report_ref_id": {"type": "string"},
        "collected_datetime": {"type": "string"}
      }
    },
    "report_dates": {
      "type": "object",
      "required": ["received_datetime","reported_datetime","partner","ref_by","lab_name"],
      "properties": {
        "ref_by": {"type": "string"},
        "partner": {"type": "string"},
        "lab_name": {"type": "string"},
        "received_datetime": {"type": "string"},
        "reported_datetime": {"type": "string"}
      }
    },
    "test_results": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["test_name","result","units","reference_interval"],
        "properties": {
          "units": {"type": "string"},
          "result": {"type": "string"},
          "test_name": {"type": "string"},
          "reference_interval": {"type": "string"}
        }
      }
    },
    "clinical_notes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["note"],
        "properties": { "note": {"type": "string"} }
      }
    }
  }
}

BASE_DIR = Path(__file__).parent.resolve()
# Vercel filesystem is read-only, use /tmp for temporary storage
if os.getenv("VERCEL"):
    UPLOAD_DIR = Path("/tmp/uploads")
    CACHE_DIR = Path("/tmp/caches")
else:
    UPLOAD_DIR = BASE_DIR / "uploads"
    CACHE_DIR = BASE_DIR / "caches"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ADE FastAPI UI")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("dpt-2-latest"),
):
    uid = uuid.uuid4().hex[:8]
    safe_name = f"{uid}_{file.filename}"
    dest_path = UPLOAD_DIR / safe_name

    try:
        with dest_path.open("wb") as out:
            _shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    # Cache key = SHA-256 of file content
    file_bytes = dest_path.read_bytes()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file_cache_dir = CACHE_DIR / file_hash
    file_cache_dir.mkdir(parents=True, exist_ok=True)
    parsed_md_path = file_cache_dir / "parsed.md"
    parsed_meta_path = file_cache_dir / "parse_meta.json"
    extracted_json_path = file_cache_dir / "extracted.json"
    schema_hash_path = file_cache_dir / "schema.hash"

    content_type = _guess_content_type(dest_path.name)

    # Parse: reuse if cached
    markdown, parse_meta = None, {}
    if parsed_md_path.exists():
        markdown = parsed_md_path.read_text(encoding="utf-8")
        if parsed_meta_path.exists():
            try:
                parse_meta = json.loads(parsed_meta_path.read_text(encoding="utf-8"))
            except Exception:
                parse_meta = {}
    else:
        try:
            with dest_path.open("rb") as f:
                parse_resp = requests.post(
                    ADE_PARSE_URL,
                    headers=HEADERS,
                    files={"document": (dest_path.name, f, content_type)},
                    data={"model": model},
                    timeout=300,
                )
            parse_resp.raise_for_status()
        except Exception as e:
            # Return error in UI instead of crashing
            print(f"Parse error: {e}")
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "file_url": f"data:{content_type};base64,{base64.b64encode(file_bytes).decode('utf-8')}",
                    "content_type": content_type,
                    "extraction": {"error": f"Parse failed: {str(e)}"},
                    "extraction_json": "{}",
                    "parse_meta": {},
                    "filename": dest_path.name,
                },
                status_code=200,
            )

        parsed = parse_resp.json()
        markdown = parsed.get("markdown") or ""
        parse_meta = parsed.get("metadata", {}) or {}
        if markdown:
            parsed_md_path.write_text(markdown, encoding="utf-8")
        parsed_meta_path.write_text(json.dumps(parse_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Extract: reuse if cached for same schema
    extraction: Dict[str, Any] = {}
    schema_str = json.dumps(SCHEMA, sort_keys=True)
    schema_hash = hashlib.sha256(schema_str.encode("utf-8")).hexdigest()

    need_extract = True
    if extracted_json_path.exists() and schema_hash_path.exists():
        try:
            existing_hash = schema_hash_path.read_text(encoding="utf-8")
            if existing_hash == schema_hash:
                cached = json.loads(extracted_json_path.read_text(encoding="utf-8"))
                extraction = cached.get("extraction", cached)
                need_extract = False
        except Exception:
            need_extract = True

    if markdown and need_extract:
        try:
            extract_resp = requests.post(
                ADE_EXTRACT_URL,
                headers=HEADERS,
                files={"markdown": ("document.md", io.BytesIO(markdown.encode("utf-8")), "text/markdown")},
                data={"schema": schema_str},
                timeout=300,
            )
            extract_resp.raise_for_status()
            result = extract_resp.json()
            print(f"DEBUG: Extract response: {json.dumps(result)}") # Debug logging
        except Exception as e:
             # Return error in UI instead of crashing
            print(f"Extract error: {e}")
            extraction = {"error": f"Extract failed: {str(e)}"}
            result = extraction # fallback

        # Cache raw provider response; UI will only render 'extraction'
        extracted_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        schema_hash_path.write_text(schema_hash, encoding="utf-8")
        extraction = result.get("extraction", result)

    elif not markdown:
        extraction = {"error": "No markdown returned from parse"}

    # Clean markdown: remove <::...::> tags
    if markdown:
        import re
        markdown = re.sub(r"<::.*?::>", "", markdown, flags=re.DOTALL)
        markdown = markdown.strip()
# In the upload function, change this section:
    # ... (previous code) ...

    # Generate Base64 Data URL for preview (bypasses Vercel ephemeral filesystem issues)
    # We already have file_bytes from line 107
    b64_content = base64.b64encode(file_bytes).decode("utf-8")
    file_data_url = f"data:{content_type};base64,{b64_content}"

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "file_url": file_data_url, # Use data URL instead of path
            "content_type": content_type,
            "extraction": extraction,
            "extraction_json": json.dumps(extraction, ensure_ascii=False, indent=2),
            "parse_meta": parse_meta,
            "markdown": markdown, # Pass markdown content
            "filename": dest_path.name,
            "error": extraction.get("error") if isinstance(extraction, dict) else None
        },
        status_code=200,
    )


def _guess_content_type(name: str) -> str:
    name_l = name.lower()
    if name_l.endswith(".pdf"):
        return "application/pdf"
    if name_l.endswith(".png"):
        return "image/png"
    if name_l.endswith(".jpg") or name_l.endswith(".jpeg"):
        return "image/jpeg"
    if name_l.endswith(".webp"):
        return "image/webp"
    if name_l.endswith(".gif"):
        return "image/gif"
    if name_l.endswith(".bmp"):
        return "image/bmp"
    return "application/octet-stream"
