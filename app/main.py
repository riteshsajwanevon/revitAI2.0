#!/usr/bin/env python3
"""
Production FastAPI for Stage 2 & Stage 3 Pipeline
Structural Beam Analysis and Column Prediction API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
import shutil
import asyncio
import time
import re
from datetime import datetime
import logging

from app.models.stage2_model import Stage2Predictor
from app.models.stage3_model import Stage3Predictor
from app.models.pipeline_manager import PipelineManager
from app.utils.file_handler import FileHandler
from app.utils.csv_processor import CSVProcessor
from app.utils.response_models import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Structural Analysis Pipeline API",
    description="Production API for Stage 2 (Column Count) and Stage 3 (Coordinate Prediction) pipelines",
    version="1.0.0"
)

# Global instances
pipeline_manager = PipelineManager()
file_handler = FileHandler()
csv_processor = CSVProcessor()

def extract_building_id_from_filename(filename: str) -> str:
    """
    Extract building ID from filename using pattern matching or generate timestamp
    
    Args:
        filename: The filename to extract building ID from
        
    Returns:
        Building ID as string (YYYYNNNN format or HHMMSS timestamp)
    """
    if not filename:
        filename = "unknown"
    
    # Check if filename matches pattern like "GMZ-2024-2323-filename"
    pattern = r'GMZ-(\d{4})-(\d+)-'
    match = re.search(pattern, filename)
    
    if match:
        year = match.group(1)
        number = match.group(2)
        return f"{year}{number}"
    else:
        # Use current timestamp as building ID
        return datetime.now().strftime("%H%M%S")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting up Structural Analysis Pipeline API...")
    await pipeline_manager.initialize_models()
    logger.info("Models loaded successfully")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Structural Analysis Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models": {
            "stage2": pipeline_manager.stage2_loaded,
            "stage3": pipeline_manager.stage3_loaded
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/stage2", response_model=Stage2Response)
async def predict_stage2(request: Stage2Request):
    """
    Stage 2: Predict column count for beams
    """
    try:
        job_id = str(uuid.uuid4())
        logger.info(f"Stage 2 prediction started - Job ID: {job_id}")
        
        # Run Stage 2 prediction
        result = await pipeline_manager.run_stage2(
            building_data=request.building_data,
            job_id=job_id
        )
        
        return Stage2Response(
            job_id=job_id,
            status="completed",
            predictions=result["predictions"],
            summary=result["summary"],
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        logger.error(f"Stage 2 prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stage 2 prediction failed: {str(e)}")

@app.post("/predict/stage3", response_model=Stage3Response)
async def predict_stage3(request: Stage3Request):
    """
    Stage 3: Predict column coordinates with Stage 2 constraints
    """
    try:
        job_id = str(uuid.uuid4())
        logger.info(f"Stage 3 prediction started - Job ID: {job_id}")
        
        # Run Stage 3 prediction
        result = await pipeline_manager.run_stage3(
            building_data=request.building_data,
            stage2_constraints=request.stage2_constraints,
            job_id=job_id
        )
        
        return Stage3Response(
            job_id=job_id,
            status="completed",
            coordinates=result["coordinates"],
            constraint_summary=result["constraint_summary"],
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        logger.error(f"Stage 3 prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stage 3 prediction failed: {str(e)}")

@app.post("/predict/pipeline", response_model=PipelineResponse)
async def predict_full_pipeline(request: PipelineRequest):
    """
    Full Pipeline: Run Stage 2 followed by constrained Stage 3
    """
    try:
        job_id = str(uuid.uuid4())
        logger.info(f"Full pipeline started - Job ID: {job_id}")
        
        # Run full pipeline
        result = await pipeline_manager.run_full_pipeline(
            building_data=request.building_data,
            job_id=job_id
        )
        
        return PipelineResponse(
            job_id=job_id,
            status="completed",
            stage2_results=result["stage2_results"],
            stage3_results=result["stage3_results"],
            total_processing_time=result["total_processing_time"]
        )
        
    except Exception as e:
        logger.error(f"Full pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Full pipeline failed: {str(e)}")

@app.post("/process-csv", response_model=CSVProcessResponse)
async def process_csv_file(
    csv_file: UploadFile = File(..., description="Single CSV file containing all building elements")
):
    """
    Process single CSV file to generate connection matrices
    This endpoint takes a single CSV with all building elements and automatically
    detects connections to generate BeamWall, BeamBeam, BeamColumn matrices and FeatureMatrix.
    Building ID is automatically generated from filename or timestamp.
    """
    try:
        # Extract building ID from filename
        filename = csv_file.filename or "unknown"
        
        # Check if filename matches pattern like "GMZ-2024-2323-filename"
        pattern = r'GMZ-(\d{4})-(\d+)-'
        match = re.search(pattern, filename)
        
        if match:
            year = match.group(1)
            number = match.group(2)
            building_id = f"{year}{number}"
        else:
            # Use current timestamp as building ID
            building_id = datetime.now().strftime("%H%M%S")
        
        job_id = str(uuid.uuid4())
        logger.info(f"CSV processing started - Job ID: {job_id}, Building: {building_id}, Filename: {filename}")
        
        # Read CSV content
        csv_content = await csv_file.read()
        csv_string = csv_content.decode('utf-8')
        
        # Process CSV to generate matrices
        result = csv_processor.process_csv(csv_string, building_id)
        
        # Store the result for potential pipeline use
        pipeline_manager.job_results[job_id] = {
            "csv_processing_results": result,
            "completed_at": datetime.now()
        }
        
        return CSVProcessResponse(
            job_id=job_id,
            status="completed",
            building_id=building_id,
            file_paths=result["file_paths"],
            connection_summary=result["connection_summary"],
            message=f"CSV processed successfully. Building ID: {building_id}. Matrices generated and ready for pipeline use.",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"CSV processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")

@app.post("/process-csv-stage2", response_model=CSVProcessWithStage2Response)
async def process_csv_with_stage2_inference(
    csv_file: UploadFile = File(..., description="Single CSV file containing all building elements")
):
    """
    Process CSV file and run Stage 2 inference
    This endpoint processes the CSV to generate matrices and immediately runs Stage 2 model
    to predict column counts for each beam, returning JSON response with model output.
    Building ID is automatically generated from filename or timestamp.
    """
    try:
        # Extract building ID from filename
        filename = csv_file.filename or "unknown"
        
        # Check if filename matches pattern like "GMZ-2024-2323-filename"
        pattern = r'GMZ-(\d{4})-(\d+)-'
        match = re.search(pattern, filename)
        
        if match:
            year = match.group(1)
            number = match.group(2)
            building_id = f"{year}{number}"
        else:
            # Use current timestamp as building ID
            building_id = datetime.now().strftime("%H%M%S")

        job_id = str(uuid.uuid4())
        start_time = time.time()
        logger.info(f"CSV processing + Stage 2 inference started - Job ID: {job_id}, Building: {building_id}, Filename: {filename}")

        # Read CSV content
        csv_content = await csv_file.read()
        csv_string = csv_content.decode('utf-8')

        # Process CSV to generate matrices
        csv_result = csv_processor.process_csv(csv_string, building_id)

        # Prepare building data for Stage 2
        building_data = {
            "building_id": building_id,
            "file_paths": csv_result["file_paths"]
        }

        # Run Stage 2 inference
        stage2_result = await pipeline_manager.run_stage2(building_data, job_id)

        total_processing_time = time.time() - start_time

        # Store complete results
        pipeline_manager.job_results[job_id] = {
            "csv_processing_results": csv_result,
            "stage2_results": stage2_result,
            "completed_at": datetime.now()
        }

        return CSVProcessWithStage2Response(
            job_id=job_id,
            status="completed",
            building_id=building_id,
            file_paths=csv_result["file_paths"],
            connection_summary=csv_result["connection_summary"],
            stage2_predictions=[
                BeamPrediction(
                    beam_id=p["beam_id"],
                    predicted_columns=p["predicted_columns"],
                    confidence=p["confidence"],
                    material_prediction=p.get("material_prediction")
                )
                for p in stage2_result["predictions"]
            ],
            stage2_summary=Stage2Summary(
                total_beams=stage2_result["summary"]["total_beams"],
                predictions_by_count=stage2_result["summary"]["predictions_by_count"],
                average_confidence=stage2_result["summary"]["average_confidence"],
                processing_time=stage2_result["summary"]["processing_time"]
            ),
            processing_time=total_processing_time,
            message=f"CSV processed and Stage 2 inference completed successfully. Building ID: {building_id}",
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"CSV processing + Stage 2 inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV processing + Stage 2 inference failed: {str(e)}")


@app.post("/process-csv-and-predict", response_model=PipelineResponse)
async def process_csv_and_run_pipeline(
    csv_file: UploadFile = File(..., description="Single CSV file containing all building elements"),
    options: Optional[Dict[str, Any]] = None
):
    """
    Process CSV file and immediately run full pipeline
    This is a convenience endpoint that combines CSV processing with pipeline execution.
    Building ID is automatically generated from filename or timestamp.
    """
    try:
        # Extract building ID from filename
        filename = csv_file.filename or "unknown"
        
        # Check if filename matches pattern like "GMZ-2024-2323-filename"
        pattern = r'GMZ-(\d{4})-(\d+)-'
        match = re.search(pattern, filename)
        
        if match:
            year = match.group(1)
            number = match.group(2)
            building_id = f"{year}{number}"
        else:
            # Use current timestamp as building ID
            building_id = datetime.now().strftime("%H%M%S")
        
        job_id = str(uuid.uuid4())
        logger.info(f"CSV processing + pipeline started - Job ID: {job_id}, Building: {building_id}, Filename: {filename}")
        
        # Read CSV content
        csv_content = await csv_file.read()
        csv_string = csv_content.decode('utf-8')
        
        # Process CSV to generate matrices
        csv_result = csv_processor.process_csv(csv_string, building_id)
        
        # Prepare building data for pipeline
        building_data = {
            "building_id": building_id,
            "file_paths": csv_result["file_paths"]
        }
        
        # Run full pipeline
        pipeline_result = await pipeline_manager.run_full_pipeline(building_data, job_id)
        
        # Add CSV processing info to response
        pipeline_result["csv_processing"] = {
            "connection_summary": csv_result["connection_summary"],
            "temp_dir": csv_result["temp_dir"]
        }
        
        return PipelineResponse(
            job_id=job_id,
            status="completed",
            stage2_results=pipeline_result["stage2_results"],
            stage3_results=pipeline_result["stage3_results"],
            total_processing_time=pipeline_result["total_processing_time"],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"CSV processing + pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV processing + pipeline failed: {str(e)}")

@app.post("/upload/building-data")
async def upload_building_data(
    feature_matrix: UploadFile = File(...),
    beam_wall_matrix: UploadFile = File(...),
    beam_beam_matrix: UploadFile = File(...),
    beam_column_matrix: Optional[UploadFile] = File(None)
):
    """
    Upload building data files for processing
    Building ID is automatically generated from the first file's filename or timestamp.
    """
    try:
        # Extract building ID from the first file's filename
        filename = feature_matrix.filename or "unknown"
        
        # Check if filename matches pattern like "GMZ-2024-2323-filename"
        pattern = r'GMZ-(\d{4})-(\d+)-'
        match = re.search(pattern, filename)
        
        if match:
            year = match.group(1)
            number = match.group(2)
            building_id = f"{year}{number}"
        else:
            # Use current timestamp as building ID
            building_id = datetime.now().strftime("%H%M%S")
        
        logger.info(f"Building data upload started - Building: {building_id}, Filename: {filename}")
        
        # Save uploaded files
        file_paths = await file_handler.save_uploaded_files(
            building_id=building_id,
            files={
                "feature_matrix": feature_matrix,
                "beam_wall_matrix": beam_wall_matrix,
                "beam_beam_matrix": beam_beam_matrix,
                "beam_column_matrix": beam_column_matrix
            }
        )
        
        return {
            "message": f"Files uploaded successfully. Building ID: {building_id}",
            "building_id": building_id,
            "files": file_paths,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """
    Get results for a specific job
    """
    try:
        results = await pipeline_manager.get_job_results(job_id)
        if not results:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get results for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")

@app.get("/download/{job_id}/{file_type}")
async def download_results(job_id: str, file_type: str):
    """
    Download result files (CSV format)
    """
    try:
        file_path = await file_handler.get_result_file(job_id, file_type)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename=f"{job_id}_{file_type}.csv",
            media_type="text/csv"
        )
        
    except Exception as e:
        logger.error(f"File download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File download failed: {str(e)}")

@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """
    Clean up temporary files for a job
    """
    try:
        await file_handler.cleanup_job_files(job_id)
        return {"message": f"Job {job_id} cleaned up successfully"}
        
    except Exception as e:
        logger.error(f"Cleanup failed for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/jobs")
async def list_jobs():
    """
    List all active jobs
    """
    try:
        jobs = await pipeline_manager.list_active_jobs()
        return {"jobs": jobs}
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)