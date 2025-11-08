"""Repository для работы с шаблонами бэктестов"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.session_manager import get_session
from backend.database.models.backtest_template import BacktestTemplate
from backend.core.logger import get_logger

logger = get_logger(__name__)


class TemplateRepository:
    """Repository для CRUD операций с шаблонами"""

    async def create(
        self,
        name: str,
        config: Dict[str, Any],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        is_public: bool = False
    ) -> BacktestTemplate:
        """Создать новый шаблон"""
        async with get_session() as session:
            template = BacktestTemplate(
                name=name,
                description=description,
                config=config,
                tags=tags or [],
                user_id=user_id,
                is_public="public" if is_public else "private"
            )

            session.add(template)
            await session.commit()
            await session.refresh(template)

            logger.info(f"✅ Template created: {template.id} ({name})")
            return template

    async def get_by_id(self, template_id: UUID) -> Optional[BacktestTemplate]:
        """Получить шаблон по ID"""
        async with get_session() as session:
            result = await session.execute(
                select(BacktestTemplate).where(BacktestTemplate.id == template_id)
            )
            return result.scalar_one_or_none()

    async def list_all(
        self,
        user_id: Optional[str] = None,
        include_public: bool = True,
        tags: Optional[List[str]] = None
    ) -> List[BacktestTemplate]:
        """Получить список шаблонов с фильтрацией"""
        async with get_session() as session:
            query = select(BacktestTemplate)

            # Фильтр по user_id и публичным шаблонам
            if user_id and include_public:
                from sqlalchemy import or_
                query = query.where(
                    or_(
                        BacktestTemplate.user_id == user_id,
                        BacktestTemplate.is_public == "public"
                    )
                )
            elif user_id:
                query = query.where(BacktestTemplate.user_id == user_id)
            elif include_public:
                query = query.where(BacktestTemplate.is_public == "public")

            # Сортировка
            query = query.order_by(BacktestTemplate.created_at.desc())

            result = await session.execute(query)
            templates = result.scalars().all()

            # Фильтр по тегам (post-processing, так как JSON)
            if tags:
                templates = [
                    t for t in templates
                    if t.tags and any(tag in t.tags for tag in tags)
                ]

            return list(templates)

    async def update(
        self,
        template_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[BacktestTemplate]:
        """Обновить шаблон"""
        async with get_session() as session:
            template = await self.get_by_id(template_id)

            if not template:
                return None

            if name is not None:
                template.name = name
            if description is not None:
                template.description = description
            if config is not None:
                template.config = config
            if tags is not None:
                template.tags = tags

            template.updated_at = datetime.utcnow()

            session.add(template)
            await session.commit()
            await session.refresh(template)

            logger.info(f"✅ Template updated: {template_id}")
            return template

    async def delete(self, template_id: UUID) -> bool:
        """Удалить шаблон"""
        async with get_session() as session:
            result = await session.execute(
                delete(BacktestTemplate).where(BacktestTemplate.id == template_id)
            )
            await session.commit()

            deleted = result.rowcount > 0
            if deleted:
                logger.info(f"🗑️  Template deleted: {template_id}")

            return deleted

    async def increment_usage(self, template_id: UUID) -> None:
        """Увеличить счетчик использования"""
        async with get_session() as session:
            template = await self.get_by_id(template_id)

            if template:
                current_count = int(template.usage_count) if template.usage_count else 0
                template.usage_count = str(current_count + 1)

                session.add(template)
                await session.commit()
