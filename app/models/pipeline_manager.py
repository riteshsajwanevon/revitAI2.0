#!/usr/bin/env python3
"""
Pipeline Manager: Orchestrates Stage 2 and Stage 3 models
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from .stage2_model import Stage2Predictor
from .stage3_model import Stage3Predictor

logger = logging.getLogger(__name__)

class PipelineManager:
    """Manages the full Stage 2 + Stage 3 pipeline"""
    
    def __init__(self):
        self.stage2_model = Stage2Predictor()
        self.stage3_model = Stage3Predictor()
        self.active_jobs = {}
        self.job_results = {}
        
    @property
    def stage2_loaded(self) -> bool:
        return self.stage2_model.loaded
    
    @property
    def stage3_loaded(self) -> bool:
        return self.stage3_model.loaded
    
    async def initialize_models(self):
        """Initialize both Stage 2 and Stage 3 models"""
        try:
            logger.info("Initializing Stage 2 model...")
            await self.stage2_model.load_model()
            
            logger.info("Initializing Stage 3 model...")
            await self.stage3_model.load_model()
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise
    
    async def run_stage2(self, building_data: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Run Stage 2 prediction only"""
        
        try:
            # Update job status
            self.active_jobs[job_id] = {
                "status": "running",
                "stage": "stage2",
                "progress": 0.0,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            # Run Stage 2 prediction
            logger.info(f"Running Stage 2 for job {job_id}")
            result = await self.stage2_model.predict(building_data)
            
            # Validate result structure
            if "predictions" not in result:
                raise ValueError("Stage 2 model did not return predictions")
            
            # Update job status
            self.active_jobs[job_id]["status"] = "completed"
            self.active_jobs[job_id]["progress"] = 1.0
            self.active_jobs[job_id]["updated_at"] = datetime.now()
            
            # Store results with enhanced structure
            self.job_results[job_id] = {
                "stage2_results": {
                    **result,
                    "job_id": job_id,
                    "status": "completed",
                    "timestamp": datetime.now()
                },
                "completed_at": datetime.now()
            }
            
            return result
            
        except Exception as e:
            # Update job status
            self.active_jobs[job_id]["status"] = "failed"
            self.active_jobs[job_id]["error_message"] = str(e)
            self.active_jobs[job_id]["updated_at"] = datetime.now()
            
            logger.error(f"Stage 2 failed for job {job_id}: {str(e)}", exc_info=True)
            raise
    
    async def run_stage3(self, building_data: Dict[str, Any], 
                        stage2_constraints: Dict[str, int], job_id: str) -> Dict[str, Any]:
        """Run Stage 3 prediction with constraints"""
        
        try:
            # Update job status
            self.active_jobs[job_id] = {
                "status": "running",
                "stage": "stage3",
                "progress": 0.0,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            # Run Stage 3 prediction
            logger.info(f"Running Stage 3 for job {job_id}")
            result = await self.stage3_model.predict(building_data, stage2_constraints)
            
            # Update job status
            self.active_jobs[job_id]["status"] = "completed"
            self.active_jobs[job_id]["progress"] = 1.0
            self.active_jobs[job_id]["updated_at"] = datetime.now()
            
            # Store results
            self.job_results[job_id] = {
                "stage3_results": result,
                "completed_at": datetime.now()
            }
            
            return result
            
        except Exception as e:
            # Update job status
            self.active_jobs[job_id]["status"] = "failed"
            self.active_jobs[job_id]["error_message"] = str(e)
            self.active_jobs[job_id]["updated_at"] = datetime.now()
            
            logger.error(f"Stage 3 failed for job {job_id}: {str(e)}")
            raise
    
    async def run_full_pipeline(self, building_data: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Run complete Stage 2 + Stage 3 pipeline"""
        
        start_time = time.time()
        
        try:
            # Update job status
            self.active_jobs[job_id] = {
                "status": "running",
                "stage": "stage2",
                "progress": 0.0,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            # Step 1: Run Stage 2
            logger.info(f"Running Stage 2 for full pipeline job {job_id}")
            stage2_results = await self.stage2_model.predict(building_data)
            
            # Validate Stage 2 results
            if "predictions" not in stage2_results:
                raise ValueError("Stage 2 model did not return predictions")
            
            logger.info(f"Stage 2 completed: {len(stage2_results['predictions'])} beam predictions, {len(stage2_results.get('column_predictions', []))} column predictions")
            
            # Update progress
            self.active_jobs[job_id]["progress"] = 0.5
            self.active_jobs[job_id]["stage"] = "stage3"
            self.active_jobs[job_id]["updated_at"] = datetime.now()
            
            # Extract constraints from Stage 2 results (only beam predictions)
            stage2_constraints = {}
            beam_predictions = [p for p in stage2_results["predictions"] if p.get("type", "beam") == "beam"]
            for prediction in beam_predictions:
                stage2_constraints[prediction["beam_id"]] = prediction["predicted_columns"]
            
            logger.info(f"Extracted {len(stage2_constraints)} beam constraints for Stage 3")
            
            # Step 2: Run Stage 3 with constraints
            logger.info(f"Running Stage 3 with constraints for job {job_id}")
            stage3_results = await self.stage3_model.predict(building_data, stage2_constraints)
            
            # Update job status
            self.active_jobs[job_id]["status"] = "completed"
            self.active_jobs[job_id]["stage"] = "complete"
            self.active_jobs[job_id]["progress"] = 1.0
            self.active_jobs[job_id]["updated_at"] = datetime.now()
            
            total_processing_time = time.time() - start_time
            
            # Combine results with enhanced Stage 2 information
            full_results = {
                "stage2_results": {
                    "job_id": job_id,
                    "status": "completed",
                    "predictions": [
                        {
                            "beam_id": p["beam_id"],
                            "predicted_columns": p["predicted_columns"],
                            "confidence": p["confidence"],
                            "predicted_material": p.get("predicted_material"),
                            "material_confidence": p.get("material_confidence"),
                            "type": p.get("type", "beam")
                        }
                        for p in stage2_results["predictions"]
                    ],
                    "column_predictions": [
                        {
                            "column_id": p["column_id"],
                            "predicted_material": p["predicted_material"],
                            "material_confidence": p["material_confidence"],
                            "material_id": p["material_id"],
                            "type": p.get("type", "column")
                        }
                        for p in stage2_results.get("column_predictions", [])
                    ],
                    "summary": {
                        "total_beams": stage2_results["summary"]["total_beams"],
                        "total_columns": stage2_results["summary"].get("total_columns", 0),
                        "predictions_by_count": stage2_results["summary"]["predictions_by_count"],
                        "average_confidence": stage2_results["summary"]["average_confidence"],
                        "average_material_confidence": stage2_results["summary"].get("average_material_confidence", 0.0),
                        "processing_time": stage2_results["summary"]["processing_time"]
                    },
                    "processing_time": stage2_results["processing_time"],
                    "timestamp": datetime.now()
                },
                "stage3_results": {
                    "job_id": job_id,
                    "status": "completed",
                    "coordinates": stage3_results["coordinates"],
                    "constraint_summary": {
                        "total_coordinates": stage3_results["constraint_summary"]["total_coordinates"],
                        "constraints_applied": stage3_results["constraint_summary"]["constraints_applied"],
                        "constraints_satisfied": stage3_results["constraint_summary"]["constraints_satisfied"],
                        "processing_time": stage3_results["constraint_summary"]["processing_time"]
                    },
                    "processing_time": stage3_results["processing_time"],
                    "timestamp": datetime.now()
                },
                "total_processing_time": total_processing_time
            }
            
            # Store results
            self.job_results[job_id] = {
                "full_pipeline_results": full_results,
                "completed_at": datetime.now()
            }
            
            return full_results
            
        except Exception as e:
            # Update job status
            self.active_jobs[job_id]["status"] = "failed"
            self.active_jobs[job_id]["error_message"] = str(e)
            self.active_jobs[job_id]["updated_at"] = datetime.now()
            
            logger.error(f"Full pipeline failed for job {job_id}: {str(e)}", exc_info=True)
            raise
    
    async def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get results for a specific job"""
        
        if job_id in self.job_results:
            return self.job_results[job_id]
        
        if job_id in self.active_jobs:
            return {
                "job_status": self.active_jobs[job_id],
                "message": "Job is still running or failed"
            }
        
        return None
    
    async def list_active_jobs(self) -> List[Dict[str, Any]]:
        """List all active jobs"""
        
        jobs = []
        for job_id, job_info in self.active_jobs.items():
            jobs.append({
                "job_id": job_id,
                **job_info
            })
        
        return jobs
    
    async def cleanup_job(self, job_id: str):
        """Clean up job data"""
        
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        if job_id in self.job_results:
            del self.job_results[job_id]
        
        logger.info(f"Cleaned up job {job_id}")
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        if job_id in self.job_results:
            return {
                "status": "completed",
                "stage": "complete",
                "progress": 1.0,
                "completed_at": self.job_results[job_id]["completed_at"]
            }
        
        return None