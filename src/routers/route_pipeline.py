from src.schema.schema import InferenceRequest, InferenceResponse
from src.pipeline import run_pipeline
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["route_pipe"])


@router.post("/infer", response_model=InferenceResponse)
def run_route_pipe(req: InferenceRequest):
    try:
        result = run_pipeline(req.video_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
