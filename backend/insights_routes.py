"""POST /api/insights - Gemini explanation of the predictions the user already received."""
import logging

from fastapi import APIRouter, Depends, HTTPException

from auth_routes import get_current_user
from insights_service import InsightsError, InsightsRequest, InsightsResponse, get_insights

log = logging.getLogger("respira")
router = APIRouter(prefix="/api/insights", tags=["insights"])


@router.post("", response_model=InsightsResponse)
def create_insights(body: InsightsRequest, user: dict = Depends(get_current_user)):
    """Sync endpoint on purpose: FastAPI runs it in a worker thread, so the blocking
    Gemini call does not freeze the server. Invalid/missing predictions -> 422 (automatic)."""
    try:
        return get_insights(body, str(user["_id"]))
    except InsightsError as e:
        log.warning("Insights failed (%s): %s", type(e).__name__, e)
        raise HTTPException(e.status_code, e.public_message)