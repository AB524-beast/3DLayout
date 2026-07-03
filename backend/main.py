from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pipeline import process_blueprint_pipeline

app = FastAPI(
    title="FloorPlan3D Vision Core Engine", 
    description="Transforms 2D structural document blueprints into volumetric layouts JSON blocks",
    version="1.0.0"
)

# Strategic CORS allowance to permit communication with Next.js browser client port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/process-layout")
async def upload_blueprint(file: UploadFile = File(...)):
    # Validate payload file integrity extensions
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file target payload architecture must be an image.")
    
    try:
        image_bytes = await file.read()
        # Feed payload into the computer vision preprocessing and parsing layout matrix
        realized_geometry_json = process_blueprint_pipeline(image_bytes)
        return realized_geometry_json
        
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Core pipeline execution error: {str(err)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)