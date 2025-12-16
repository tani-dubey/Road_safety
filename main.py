from src.pipeline import main
from fastapi import FastAPI
from src.routers import route_metrices,route_health,route_pipeline

app= FastAPI(
    title= "Number plate Analytics API",
    version="0.1.0",
    description="Analytics and inference service for vechile number plates"
)

@app.get('/')
def home():
    return {
        "status": "Ok"
    }

app.include_router(route_metrices.router, prefix="/api/v1")
app.include_router(route_health.router, prefix="/api/v1")
app.include_router(route_pipeline.router, prefix="/api/v1")


if __name__=="__main__":
    main()