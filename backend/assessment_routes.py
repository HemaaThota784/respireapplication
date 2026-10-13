"""Per-user assessment history stored in MongoDB."""
from datetime import datetime, timezone
from typing import Dict, Optional

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from auth_routes import get_current_user
from db import get_db

router = APIRouter(prefix="/api/assessments", tags=["assessments"])


class PredictionIn(BaseModel):
    predicted_class: str = Field(max_length=60)
    label: str = Field(max_length=60)
    confidence: float = Field(ge=0, le=100)
    probabilities: Dict[str, float] = Field(max_length=10)
    extra: Optional[str] = Field(default=None, max_length=200)


class AssessmentIn(BaseModel):
    audio: Optional[PredictionIn] = None
    xray: Optional[PredictionIn] = None

    @model_validator(mode="after")
    def need_one(self):
        if self.audio is None and self.xray is None:
            raise ValueError("Provide an audio and/or X-ray result.")
        return self


def _public(a: dict) -> dict:
    return {"id": str(a["_id"]), "created_at": a["created_at"].isoformat(),
            "audio": a.get("audio"), "xray": a.get("xray")}


@router.post("", status_code=201)
def create_assessment(body: AssessmentIn, user: dict = Depends(get_current_user)):
    doc = {"user_id": user["_id"], "created_at": datetime.now(timezone.utc),
           "audio": body.audio.model_dump() if body.audio else None,
           "xray": body.xray.model_dump() if body.xray else None}
    doc["_id"] = get_db().assessments.insert_one(doc).inserted_id
    return _public(doc)


@router.get("")
def list_assessments(limit: int = Query(50, ge=1, le=200), user: dict = Depends(get_current_user)):
    cur = get_db().assessments.find({"user_id": user["_id"]}).sort("created_at", -1).limit(limit)
    return [_public(a) for a in cur]


@router.delete("/{assessment_id}")
def delete_assessment(assessment_id: str, user: dict = Depends(get_current_user)):
    try:
        oid = ObjectId(assessment_id)
    except InvalidId:
        raise HTTPException(404, "Assessment not found.")
    res = get_db().assessments.delete_one({"_id": oid, "user_id": user["_id"]})
    if res.deleted_count == 0:
        raise HTTPException(404, "Assessment not found.")
    return {"ok": True}
