import json
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from services.parsers import ParserRegistry, ExtractionResult
from services.task_service import task_service, TaskRecord
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
        "parsers": [p.parser_id for p in registry.parsers] + [registry.gemini_extractor.parser_id],
        "active_tasks": len(task_service.list_tasks(limit=10))
    }

@router.post("/extract", response_model=ExtractionResult)
async def api_extract_document(
    file: UploadFile = File(...),
    doc_type: str = Form("unknown"),
    hints: Optional[str] = Form("{}")
):
    filename = file.filename or "unknown_file"
    task = task_service.create_task(
        task_type="financial_extraction",
        source="REST API",
        name=filename,
        summary=f"Extracting structured data (hint: {doc_type})"
    )

    try:
        file_bytes = await file.read()
        max_size = config.config.MAX_EXTRACTION_FILE_SIZE
        if len(file_bytes) > max_size:
            err_msg = f"File exceeds maximum allowed size of {max_size / (1024*1024):.1f}MB"
            task_service.fail_task(task.id, error=err_msg)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=err_msg
            )

        parsed_hints: Dict[str, Any] = {}
        if hints:
            try:
                parsed_hints = json.loads(hints)
            except Exception:
                parsed_hints = {"raw_hints": hints}

        result = registry.extract(
            file_bytes=file_bytes,
            filename=filename,
            doc_type=doc_type,
            hints=parsed_hints
        )

        if result.success:
            summary_msg = f"Extracted {len(result.transactions)} transactions, {len(result.assets)} assets ({result.template_id})"
            task_service.complete_task(
                task_id=task.id,
                summary=summary_msg,
                output=result.model_dump(),
                duration_ms=result.processing_time_ms
            )
        else:
            task_service.fail_task(
                task_id=task.id,
                error=result.error or "Unknown extraction error",
                duration_ms=result.processing_time_ms
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during document extraction: {str(e)}")
        task_service.fail_task(task.id, error=str(e))
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

@router.get("/tasks")
def api_list_tasks(limit: int = 50, status: Optional[str] = None, source: Optional[str] = None):
    try:
        tasks = task_service.list_tasks(limit=limit, status_filter=status, source_filter=source)
        return {
            "tasks": [t.model_dump() for t in tasks],
            "count": len(tasks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}")
def api_get_task(task_id: str):
    task = task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task.model_dump()

