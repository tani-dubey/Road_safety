from fastapi import APIRouter

router= APIRouter(tags=["metrics"])

@router.get('/metrics')
def get_metrics():
    return {
        "supported_models": ["YOLOv8"],
        "Cpu_interface": True,
        "Openvino_ready": False
    }