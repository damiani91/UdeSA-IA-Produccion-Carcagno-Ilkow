# src/api/routes/wells.py
import pandas as pd
from fastapi import APIRouter, Query
from src.api import _state
from src.api.schemas import WellInfo

router = APIRouter()

@router.get("/wells", response_model=list[WellInfo])
def get_wells(date_query: str = Query(...)):
    df = _state.features_df
    active = df[df["fecha"] <= pd.Timestamp(date_query)]["idpozo"].unique()
    return [WellInfo(id_well=str(w)) for w in sorted(active)]