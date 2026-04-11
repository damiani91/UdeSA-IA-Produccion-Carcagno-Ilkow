# src/api/routes/wells.py
from fastapi import APIRouter, Query, Request
from src.api.schemas import WellInfo

router = APIRouter()

@router.get("/wells", response_model=list[WellInfo])
def get_wells(request: Request, date_query: str = Query(...)):
    import pandas as pd
    df = request.app.state.features_df
    active = df[df["fecha"] <= pd.Timestamp(date_query)]["idpozo"].unique()
    return [WellInfo(id_well=str(w)) for w in sorted(active)]