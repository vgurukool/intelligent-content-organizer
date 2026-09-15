import json
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from services.parsers import ParserRegistry, ExtractionResult
import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Extraction"])

# Singleton registry instance
registry = ParserRegistry()

@router.get("/health")
def api_health():
    return {
        "status": "healthy",
        "service": "incorg-extraction-engine",
        "templates_loaded": len(registry.list_templates()),
        "parsers": [p.parser_id for p in registry.parsers] + [registry.gemini_extractor.parser_id]
    }

@router.post("/extract", response_model=ExtractionResult)
async def api_extract_document(
    file: UploadFile = File(...),
    doc_type: str = Form("unknown"),
    hints: Optional[str] = Form("{}")
):
    try:
        file_bytes = await file.read()
        max_size = config.config.MAX_EXTRACTION_FILE_SIZE
        if len(file_bytes) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {max_size / (1024*1024):.1f}MB"
            )

        parsed_hints: Dict[str, Any] = {}
        if hints:
            try:
                parsed_hints = json.loads(hints)
            except Exception:
                parsed_hints = {"raw_hints": hints}

        filename = file.filename or "unknown_file"
        result = registry.extract(
            file_bytes=file_bytes,
            filename=filename,
            doc_type=doc_type,
            hints=parsed_hints
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during document extraction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}"
        )

@router.get("/templates")
def api_list_templates():
    try:
        templates = registry.list_templates()
        return {"templates": templates, "count": len(templates)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates/{template_id}")
def api_get_template(template_id: str):
    tpl = registry.get_template(template_id)
    if not tpl:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
    return tpl
