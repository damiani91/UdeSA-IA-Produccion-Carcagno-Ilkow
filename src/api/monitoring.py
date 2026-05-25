# src/api/monitoring.py
"""Endpoint de monitoreo: devuelve el reporte de drift+decay vigente."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.monitoring.report import load_latest_report, load_report_by_date

router = APIRouter()


@router.get("/monitoring/report")
def get_drift_report(date: Optional[str] = Query(None, description="YYYY-MM-DD; default: último reporte disponible")):
    """Devuelve el reporte de drift y decay para `date`. Si no se pasa, el más reciente."""
    report = load_report_by_date(date) if date else load_latest_report()
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="No hay reportes de drift disponibles. Corré el DAG de Airflow primero.",
        )
    return report
