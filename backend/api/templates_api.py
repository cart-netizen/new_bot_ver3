"""API endpoints для управления шаблонами бэктестов"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID

from backend.core.logger import get_logger
from backend.infrastructure.repositories.templates.template_repository import TemplateRepository

logger = get_logger(__name__)

router = APIRouter(prefix="/api/backtesting/templates", tags=["Templates"])

# Repository
template_repo = TemplateRepository()


# Request/Response Models
class CreateTemplateRequest(BaseModel):
    """Request для создания шаблона"""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    config: Dict[str, Any] = Field(...)
    tags: List[str] = Field(default_factory=list)
    is_public: bool = Field(default=False)


class UpdateTemplateRequest(BaseModel):
    """Request для обновления шаблона"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class TemplateResponse(BaseModel):
    """Response с данными шаблона"""
    id: str
    name: str
    description: Optional[str]
    config: Dict[str, Any]
    tags: List[str]
    is_public: bool
    usage_count: int
    created_at: datetime
    updated_at: datetime


# Endpoints
@router.post("")
async def create_template(request: CreateTemplateRequest) -> Dict[str, Any]:
    """Создать новый шаблон"""
    try:
        template = await template_repo.create(
            name=request.name,
            description=request.description,
            config=request.config,
            tags=request.tags,
            is_public=request.is_public
        )

        return {
            "success": True,
            "template_id": str(template.id),
            "message": f"Template '{request.name}' created successfully"
        }

    except Exception as e:
        logger.error(f"Error creating template: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_templates(
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    include_public: bool = Query(True, description="Include public templates")
) -> Dict[str, Any]:
    """Получить список шаблонов"""
    try:
        tag_list = tags.split(",") if tags else None

        templates = await template_repo.list_all(
            include_public=include_public,
            tags=tag_list
        )

        return {
            "templates": [
                {
                    "id": str(t.id),
                    "name": t.name,
                    "description": t.description,
                    "config": t.config,
                    "tags": t.tags or [],
                    "is_public": t.is_public == "public",
                    "usage_count": int(t.usage_count) if t.usage_count else 0,
                    "created_at": t.created_at.isoformat(),
                    "updated_at": t.updated_at.isoformat()
                }
                for t in templates
            ],
            "total": len(templates)
        }

    except Exception as e:
        logger.error(f"Error listing templates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{template_id}")
async def get_template(template_id: UUID) -> Dict[str, Any]:
    """Получить шаблон по ID"""
    try:
        template = await template_repo.get_by_id(template_id)

        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

        # Increment usage counter
        await template_repo.increment_usage(template_id)

        return {
            "id": str(template.id),
            "name": template.name,
            "description": template.description,
            "config": template.config,
            "tags": template.tags or [],
            "is_public": template.is_public == "public",
            "usage_count": int(template.usage_count) if template.usage_count else 0,
            "created_at": template.created_at.isoformat(),
            "updated_at": template.updated_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template {template_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{template_id}")
async def update_template(
    template_id: UUID,
    request: UpdateTemplateRequest
) -> Dict[str, Any]:
    """Обновить шаблон"""
    try:
        template = await template_repo.update(
            template_id=template_id,
            name=request.name,
            description=request.description,
            config=request.config,
            tags=request.tags
        )

        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

        return {
            "success": True,
            "template_id": str(template.id),
            "message": "Template updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating template {template_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{template_id}")
async def delete_template(template_id: UUID) -> Dict[str, Any]:
    """Удалить шаблон"""
    try:
        deleted = await template_repo.delete(template_id)

        if not deleted:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

        return {
            "success": True,
            "template_id": str(template_id),
            "message": "Template deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting template {template_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
