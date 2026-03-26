#!/usr/bin/env python3
"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

# ============================================================================
# Request Models
# ============================================================================

class BuildingData(BaseModel):
    """Building data structure"""
    building_id: str
    feature_matrix: Optional[List[Dict[str, Any]]] = None
    beam_wall_matrix: Optional[List[Dict[str, Any]]] = None
    beam_beam_matrix: Optional[List[Dict[str, Any]]] = None
    beam_column_matrix: Optional[List[Dict[str, Any]]] = None
    file_paths: Optional[Dict[str, str]] = None

class Stage2Request(BaseModel):
    """Stage 2 prediction request"""
    building_data: BuildingData
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

class Stage3Request(BaseModel):
    """Stage 3 prediction request"""
    building_data: BuildingData
    stage2_constraints: Dict[str, int]  # beam_id -> max_columns
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

class PipelineRequest(BaseModel):
    """Full pipeline request"""
    building_data: BuildingData
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

# ============================================================================
# Response Models
# ============================================================================

class ColumnPrediction(BaseModel):
    """Individual column material prediction"""
    column_id: str
    predicted_material: str
    material_confidence: Optional[float] = None
    material_id: Optional[int] = None
    type: Optional[str] = "column"

class BeamPrediction(BaseModel):
    """Individual beam prediction"""
    beam_id: str
    predicted_columns: int
    confidence: Optional[float] = None
    predicted_material: Optional[str] = None
    material_confidence: Optional[float] = None
    predicted_column_length: Optional[float] = None
    type: Optional[str] = "beam"

class ColumnCoordinate(BaseModel):
    """Column coordinate prediction"""
    building_id: str
    beam_id: str
    column_id: str
    x: float
    y: float
    z: float
    z_min: Optional[float] = None
    confidence: Optional[float] = None

class Stage2Summary(BaseModel):
    """Stage 2 prediction summary"""
    total_beams: int
    total_columns: Optional[int] = 0
    predictions_by_count: Dict[str, int]
    average_confidence: Optional[float] = None
    average_material_confidence: Optional[float] = None
    average_predicted_column_length: Optional[float] = None
    processing_time: float

class Stage3Summary(BaseModel):
    """Stage 3 prediction summary"""
    total_coordinates: int
    constraints_applied: int
    constraints_satisfied: int
    average_confidence: Optional[float] = None
    processing_time: float

class Stage2Response(BaseModel):
    """Stage 2 API response"""
    job_id: str
    status: str
    predictions: List[BeamPrediction]
    column_predictions: Optional[List[ColumnPrediction]] = []
    summary: Stage2Summary
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

class Stage3Response(BaseModel):
    """Stage 3 API response"""
    job_id: str
    status: str
    coordinates: List[ColumnCoordinate]
    constraint_summary: Stage3Summary
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

class PipelineResponse(BaseModel):
    """Full pipeline API response"""
    job_id: str
    status: str
    stage2_results: Stage2Response
    stage3_results: Stage3Response
    total_processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str
    job_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# ============================================================================
# Status Models
# ============================================================================

class JobStatus(BaseModel):
    """Job status model"""
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    stage: Optional[str] = None  # "stage2", "stage3", "complete"
    progress: Optional[float] = None  # 0.0 to 1.0
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None

class HealthStatus(BaseModel):
    """API health status"""
    status: str
    models: Dict[str, bool]
    timestamp: datetime
    uptime: Optional[float] = None

# ============================================================================
# File Upload Models
# ============================================================================

class FileUploadResponse(BaseModel):
    """File upload response"""
    message: str
    building_id: str
    files: Dict[str, str]  # file_type -> file_path
    timestamp: datetime = Field(default_factory=datetime.now)

class CSVProcessResponse(BaseModel):
    """CSV processing response"""
    job_id: str
    status: str
    building_id: str
    file_paths: Dict[str, str]  # Generated matrix file paths
    connection_summary: Dict[str, int]  # Connection statistics
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class CSVProcessWithStage2Response(BaseModel):
    """CSV processing with Stage 2 inference response"""
    job_id: str
    status: str
    building_id: str
    file_paths: Dict[str, str]  # Generated matrix file paths
    connection_summary: Dict[str, int]  # Connection statistics
    stage2_predictions: List[BeamPrediction]  # Stage 2 model output
    stage2_summary: Stage2Summary  # Stage 2 summary statistics
    processing_time: float
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class FileDownloadInfo(BaseModel):
    """File download information"""
    job_id: str
    file_type: str
    file_path: str
    file_size: Optional[int] = None
    created_at: datetime