import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pipeline import process_blueprint_pipeline

app = FastAPI(
    title="3D Layout Pipeline API",
    description="Universal 2D Blueprint to 3D Structured Layout Converter Engine",
    version="1.0.0"
)

# --- CORS MIDDLEWARE SYSTEM ---
# Crucial for allowing your React/Next.js frontend (running on a different port like 3000)
# to securely transmit drag-and-drop file streams to this backend server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your specific frontend URL (e.g. ["http://localhost:3000"]) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/process-layout")
async def process_layout(file: UploadFile = File(...)):
    """
    Accepts raw multipart form image files dropped or browsed via the frontend,
    reads them asynchronously, and extracts full structured 3D spatial definitions.
    """
    # 1. Validate MIME type explicitly for incoming drag-and-drop files
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type format: '{file.content_type}'. Please drop a valid JPEG or PNG blueprint asset."
        )
    
    try:
        # 2. Read file contents directly as raw binary streams
        image_bytes = await file.read()
        
        # 3. Process the file using your universal topological contour pipeline
        structured_layout_json = process_blueprint_pipeline(image_bytes)
        
        # 4. Return clean, JSON-serializable native dict structures to the client
        return structured_layout_json

    except ValueError as val_err:
        # Catch explicit image decoding format errors from OpenCV
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as e:
        # Catch unexpected pipeline runtime issues gracefully
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")


@app.get("/api/v1/health")
def health_check():
    """Simple connection gate to verify backend service status."""
    return {"status": "healthy", "engine": "Universal Blueprint Topology Parser v1"}


if __name__ == "__main__":
    # Start local server via script execution (python main.py)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)