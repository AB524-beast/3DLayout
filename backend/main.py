from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os

# Import the pipeline - handle import errors gracefully
try:
    from pipeline import process_blueprint_pipeline
except ImportError as e:
    print(f"WARNING: Could not import pipeline: {e}")
    process_blueprint_pipeline = None

app = FastAPI(
    title="FloorPlan3D Vision Core Engine", 
    description="Transforms 2D structural document blueprints into volumetric layouts JSON blocks",
    version="1.0.1"
)

# CORS: Allow all origins for development, restrict in production
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify backend is running."""
    status = "ok" if process_blueprint_pipeline else "degraded"
    return {"status": status, "version": "1.0.1"}

@app.post("/api/v1/process-layout")
async def upload_blueprint(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (PNG, JPG, JPEG).")
    
    if not process_blueprint_pipeline:
        raise HTTPException(status_code=503, detail="CV pipeline not available. Check server logs.")
    
    try:
        image_bytes = await file.read()
        
        # Validate image size (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large. Max size: 10MB.")
        
        # Process through CV pipeline
        result = process_blueprint_pipeline(image_bytes)
        
        # Validate result has rooms
        if not result.get("rooms"):
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Could not detect room structures in image.",
                    "rooms": [{
                        "label": "Default Room",
                        "dimensions": "6.0m x 4.5m",
                        "centerX": 0.0,
                        "centerY": 0.0,
                        "walls": [
                            {"x1": -6, "y1": -4.5, "x2": 6, "y2": -4.5},
                            {"x1": 6, "y1": -4.5, "x2": 6, "y2": 4.5},
                            {"x1": 6, "y1": 4.5, "x2": -6, "y2": 4.5},
                            {"x1": -6, "y1": 4.5, "x2": -6, "y2": -4.5}
                        ]
                    }]
                }
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as err:
        import traceback
        print(f"Pipeline error: {err}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(err)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)