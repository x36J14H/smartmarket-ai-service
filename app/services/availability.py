"""
Обратная совместимость — перенаправляем в onec_client.

Всё реальное API к 1С теперь в app/services/onec_client.py.
"""
from app.services.onec_client import fetch_availability, filter_available_ids

__all__ = ["fetch_availability", "filter_available_ids"]
