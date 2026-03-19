#!/usr/bin/env python3
"""
File Handler: Manages file uploads, downloads, and temporary storage
"""

import os
import shutil
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from fastapi import UploadFile
import uuid
from datetime import datetime
import aiofiles

logger = logging.getLogger(__name__)

class FileHandler:
    """Handles file operations for the API"""
    
    def __init__(self, base_dir: str = "temp_data"):
        self.base_dir = base_dir
        self.upload_dir = os.path.join(base_dir, "uploads")
        self.results_dir = os.path.join(base_dir, "results")
        
        # Create directories
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    async def save_uploaded_files(self, building_id: str, 
                                 files: Dict[str, UploadFile]) -> Dict[str, str]:
        """Save uploaded files and return file paths"""
        
        file_paths = {}
        building_dir = os.path.join(self.upload_dir, building_id)
        os.makedirs(building_dir, exist_ok=True)
        
        try:
            for file_type, upload_file in files.items():
                if upload_file is None:
                    continue
                
                # Generate filename
                filename = f"{building_id}_{file_type}.csv"
                file_path = os.path.join(building_dir, filename)
                
                # Save file
                async with aiofiles.open(file_path, 'wb') as f:
                    content = await upload_file.read()
                    await f.write(content)
                
                file_paths[file_type] = file_path
                logger.info(f"Saved {file_type} to {file_path}")
            
            return file_paths
            
        except Exception as e:
            logger.error(f"Failed to save uploaded files: {str(e)}")
            # Cleanup on failure
            if os.path.exists(building_dir):
                shutil.rmtree(building_dir)
            raise
    
    async def save_results(self, job_id: str, results: Dict[str, Any], 
                          result_type: str) -> Dict[str, str]:
        """Save prediction results to CSV files"""
        
        job_dir = os.path.join(self.results_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        saved_files = {}
        
        try:
            if result_type == "stage2":
                # Save Stage 2 predictions
                predictions_df = pd.DataFrame(results["predictions"])
                predictions_file = os.path.join(job_dir, "stage2_predictions.csv")
                predictions_df.to_csv(predictions_file, index=False)
                saved_files["predictions"] = predictions_file
                
                # Save summary
                summary_df = pd.DataFrame([results["summary"]])
                summary_file = os.path.join(job_dir, "stage2_summary.csv")
                summary_df.to_csv(summary_file, index=False)
                saved_files["summary"] = summary_file
                
            elif result_type == "stage3":
                # Save Stage 3 coordinates
                coordinates_df = pd.DataFrame(results["coordinates"])
                coordinates_file = os.path.join(job_dir, "stage3_coordinates.csv")
                coordinates_df.to_csv(coordinates_file, index=False)
                saved_files["coordinates"] = coordinates_file
                
                # Save constraint summary
                summary_df = pd.DataFrame([results["constraint_summary"]])
                summary_file = os.path.join(job_dir, "stage3_constraint_summary.csv")
                summary_df.to_csv(summary_file, index=False)
                saved_files["constraint_summary"] = summary_file
                
            elif result_type == "pipeline":
                # Save full pipeline results
                
                # Stage 2 results
                stage2_predictions = pd.DataFrame(results["stage2_results"]["predictions"])
                stage2_file = os.path.join(job_dir, "pipeline_stage2_predictions.csv")
                stage2_predictions.to_csv(stage2_file, index=False)
                saved_files["stage2_predictions"] = stage2_file
                
                # Stage 3 results
                stage3_coordinates = pd.DataFrame(results["stage3_results"]["coordinates"])
                stage3_file = os.path.join(job_dir, "pipeline_stage3_coordinates.csv")
                stage3_coordinates.to_csv(stage3_file, index=False)
                saved_files["stage3_coordinates"] = stage3_file
                
                # Combined summary
                pipeline_summary = {
                    "job_id": job_id,
                    "total_beams": results["stage2_results"]["summary"]["total_beams"],
                    "total_coordinates": results["stage3_results"]["constraint_summary"]["total_coordinates"],
                    "constraints_applied": results["stage3_results"]["constraint_summary"]["constraints_applied"],
                    "constraints_satisfied": results["stage3_results"]["constraint_summary"]["constraints_satisfied"],
                    "stage2_processing_time": results["stage2_results"]["processing_time"],
                    "stage3_processing_time": results["stage3_results"]["processing_time"],
                    "total_processing_time": results["total_processing_time"]
                }
                
                summary_df = pd.DataFrame([pipeline_summary])
                summary_file = os.path.join(job_dir, "pipeline_summary.csv")
                summary_df.to_csv(summary_file, index=False)
                saved_files["pipeline_summary"] = summary_file
            
            logger.info(f"Saved {result_type} results for job {job_id}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Failed to save results for job {job_id}: {str(e)}")
            raise
    
    async def get_result_file(self, job_id: str, file_type: str) -> Optional[str]:
        """Get path to a specific result file"""
        
        job_dir = os.path.join(self.results_dir, job_id)
        
        # Map file types to actual filenames
        file_mapping = {
            "stage2_predictions": "stage2_predictions.csv",
            "stage2_summary": "stage2_summary.csv",
            "stage3_coordinates": "stage3_coordinates.csv",
            "stage3_constraint_summary": "stage3_constraint_summary.csv",
            "pipeline_stage2_predictions": "pipeline_stage2_predictions.csv",
            "pipeline_stage3_coordinates": "pipeline_stage3_coordinates.csv",
            "pipeline_summary": "pipeline_summary.csv"
        }
        
        if file_type not in file_mapping:
            return None
        
        file_path = os.path.join(job_dir, file_mapping[file_type])
        
        if os.path.exists(file_path):
            return file_path
        
        return None
    
    async def cleanup_job_files(self, job_id: str):
        """Clean up all files for a specific job"""
        
        # Clean up upload files
        upload_job_dir = os.path.join(self.upload_dir, job_id)
        if os.path.exists(upload_job_dir):
            shutil.rmtree(upload_job_dir)
            logger.info(f"Cleaned up upload files for job {job_id}")
        
        # Clean up result files
        result_job_dir = os.path.join(self.results_dir, job_id)
        if os.path.exists(result_job_dir):
            shutil.rmtree(result_job_dir)
            logger.info(f"Cleaned up result files for job {job_id}")
    
    async def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up files older than specified hours"""
        
        current_time = datetime.now()
        cleaned_count = 0
        
        # Clean upload files
        for item in os.listdir(self.upload_dir):
            item_path = os.path.join(self.upload_dir, item)
            if os.path.isdir(item_path):
                # Check directory modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(item_path))
                age_hours = (current_time - mod_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    shutil.rmtree(item_path)
                    cleaned_count += 1
                    logger.info(f"Cleaned up old upload directory: {item}")
        
        # Clean result files
        for item in os.listdir(self.results_dir):
            item_path = os.path.join(self.results_dir, item)
            if os.path.isdir(item_path):
                # Check directory modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(item_path))
                age_hours = (current_time - mod_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    shutil.rmtree(item_path)
                    cleaned_count += 1
                    logger.info(f"Cleaned up old result directory: {item}")
        
        logger.info(f"Cleaned up {cleaned_count} old directories")
        return cleaned_count
    
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a file"""
        
        if not os.path.exists(file_path):
            return None
        
        stat = os.stat(file_path)
        
        return {
            "file_path": file_path,
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime),
            "modified_at": datetime.fromtimestamp(stat.st_mtime)
        }
    
    async def validate_csv_file(self, file_path: str, required_columns: List[str] = None) -> bool:
        """Validate CSV file format"""
        
        try:
            df = pd.read_csv(file_path)
            
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    logger.warning(f"Missing columns in {file_path}: {missing_columns}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate CSV file {file_path}: {str(e)}")
            return False