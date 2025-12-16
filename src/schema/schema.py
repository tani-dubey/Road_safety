from pydantic import BaseModel
from typing import List

class InferenceRequest(BaseModel):
    video_path: str

class InferenceResponse(BaseModel):
    total_frames: int
    total_vehicles: int
    output_files: List[str]
