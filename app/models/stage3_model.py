#!/usr/bin/env python3
"""
Stage 3 Model: Column Coordinate Prediction with Constraints
Production wrapper for Stage 3 CNN
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy.signal import find_peaks
import time

logger = logging.getLogger(__name__)

class ImprovedColumnPredictor(nn.Module):
    """1D CNN for column coordinate prediction"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 1, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(self.relu(self.bn4(self.conv4(x))))
        x = self.sigmoid(self.conv5(x))
        return x.squeeze(1)

class Stage3Predictor:
    """Production wrapper for Stage 3 model"""
    
    def __init__(self, model_path: str = "../column_predictor_no_leakage.pth"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.peak_threshold = 0.35
        self.segment_size = 0.5
        self.loaded = False
        
    async def load_model(self):
        """Load the trained Stage 3 model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Initialize model
            self.model = ImprovedColumnPredictor()
            
            # Load state dict with numpy compatibility fix
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
            except ModuleNotFoundError as e:
                if "numpy._core" in str(e):
                    # Handle numpy version compatibility issue
                    import sys
                    import numpy
                    # Create compatibility alias
                    if not hasattr(numpy, '_core'):
                        numpy._core = numpy.core
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                else:
                    raise
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Checkpoint contains metadata, extract model state dict
                state_dict = checkpoint["model_state_dict"]
                logger.info("Loading model from checkpoint with metadata")
            else:
                # Direct state dict (OrderedDict or dict)
                state_dict = checkpoint
                logger.info("Loading model from direct state dict")
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            logger.info(f"Stage 3 model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load Stage 3 model: {str(e)}")
            raise
    
    def _calc_beam_length(self, beam_row: pd.Series) -> float:
        """Calculate beam length"""
        sx, sy, sz = beam_row["Start X"], beam_row["Start Y"], beam_row["Start Z"]
        ex, ey, ez = beam_row["End X"], beam_row["End Y"], beam_row["End Z"]
        return np.sqrt((ex-sx)**2 + (ey-sy)**2 + (ez-sz)**2)
    
    def _project_point_on_beam(self, point_xyz: np.ndarray, beam_start: np.ndarray, 
                              beam_end: np.ndarray) -> float:
        """Project point onto beam axis and return relative position (0-1)"""
        beam_vec = beam_end - beam_start
        length_sq = np.dot(beam_vec, beam_vec)
        if length_sq < 1e-6:
            return 0.0
        t = np.dot(point_xyz - beam_start, beam_vec) / length_sq
        return float(np.clip(t, 0.0, 1.0))
    
    def _generate_beam_signals(self, feature_df: pd.DataFrame, beam_wall_df: pd.DataFrame,
                              beam_beam_df: pd.DataFrame, beam_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate wall and beam signals for a specific beam"""
        
        # Get beam info
        beam_info = feature_df[
            (feature_df["Element Type"] == "Structural Framing") &
            (feature_df["Element ID"].astype(str).str.contains(beam_id))
        ]
        
        if beam_info.empty:
            raise ValueError(f"Beam {beam_id} not found in FeatureMatrix")
        
        beam_row = beam_info.iloc[0]
        beam_length = self._calc_beam_length(beam_row)
        
        # Create beam axis
        beam_start = np.array([beam_row["Start X"], beam_row["Start Y"], beam_row["Start Z"]])
        beam_end = np.array([beam_row["End X"], beam_row["End Y"], beam_row["End Z"]])
        
        # Initialize signals
        num_segments = int(beam_length / self.segment_size) + 1
        wall_signal = np.zeros(num_segments)
        beam_signal = np.zeros(num_segments)
        
        # Generate wall signal
        try:
            beam_wall_row = beam_wall_df[beam_wall_df.index.astype(str).str.contains(beam_id)]
            if not beam_wall_row.empty:
                connected_walls = beam_wall_row.iloc[0]
                wall_connections = connected_walls[connected_walls == 1].index.tolist()
                
                for wall_id in wall_connections:
                    # Get wall info
                    wall_info = feature_df[
                        (feature_df["Element Type"] == "Wall") &
                        (feature_df["Element ID"].astype(str).str.contains(str(wall_id)))
                    ]
                    
                    if not wall_info.empty:
                        wall_row = wall_info.iloc[0]
                        wall_center = np.array([
                            (wall_row["Start X"] + wall_row["End X"]) / 2,
                            (wall_row["Start Y"] + wall_row["End Y"]) / 2,
                            (wall_row["Start Z"] + wall_row["End Z"]) / 2
                        ])
                        
                        # Project wall center onto beam
                        rel_pos = self._project_point_on_beam(wall_center, beam_start, beam_end)
                        segment_idx = int(rel_pos * (num_segments - 1))
                        wall_signal[segment_idx] = 1.0
        except Exception as e:
            logger.warning(f"Failed to generate wall signal for beam {beam_id}: {str(e)}")
        
        # Generate beam signal
        try:
            beam_beam_row = beam_beam_df[beam_beam_df.index.astype(str).str.contains(beam_id)]
            if not beam_beam_row.empty:
                connected_beams = beam_beam_row.iloc[0]
                beam_connections = connected_beams[connected_beams == 1].index.tolist()
                
                for other_beam_id in beam_connections:
                    if str(other_beam_id) == beam_id:
                        continue
                    
                    # Get other beam info
                    other_beam_info = feature_df[
                        (feature_df["Element Type"] == "Structural Framing") &
                        (feature_df["Element ID"].astype(str).str.contains(str(other_beam_id)))
                    ]
                    
                    if not other_beam_info.empty:
                        other_beam_row = other_beam_info.iloc[0]
                        other_beam_center = np.array([
                            (other_beam_row["Start X"] + other_beam_row["End X"]) / 2,
                            (other_beam_row["Start Y"] + other_beam_row["End Y"]) / 2,
                            (other_beam_row["Start Z"] + other_beam_row["End Z"]) / 2
                        ])
                        
                        # Project other beam center onto current beam
                        rel_pos = self._project_point_on_beam(other_beam_center, beam_start, beam_end)
                        segment_idx = int(rel_pos * (num_segments - 1))
                        beam_signal[segment_idx] = 1.0
        except Exception as e:
            logger.warning(f"Failed to generate beam signal for beam {beam_id}: {str(e)}")
        
        return wall_signal, beam_signal
    
    def _extract_constrained_peaks(self, signal: np.ndarray, max_peaks: int) -> np.ndarray:
        """Extract peaks from signal, constrained by Stage 2 prediction"""
        
        if max_peaks == 0:
            return np.array([])
        
        # Find all potential peaks
        all_peaks, _ = find_peaks(signal, height=self.peak_threshold, distance=5)
        
        # Include edges if they meet threshold
        edge_peaks = []
        if len(signal) > 0:
            if signal[0] >= self.peak_threshold:
                edge_peaks.append(0)
            if signal[-1] >= self.peak_threshold:
                edge_peaks.append(len(signal)-1)
        
        # Combine all peaks
        if len(edge_peaks) > 0:
            all_peaks = np.concatenate([edge_peaks, all_peaks])
        all_peaks = np.unique(all_peaks)
        
        # If we have fewer or equal peaks than max_peaks, return all
        if len(all_peaks) <= max_peaks:
            return all_peaks
        
        # Select strongest peaks
        peak_heights = signal[all_peaks]
        strongest_indices = np.argsort(peak_heights)[-max_peaks:]
        selected_peaks = all_peaks[strongest_indices]
        selected_peaks.sort()
        
        return selected_peaks
    
    def _convert_peaks_to_coordinates(self, peaks: np.ndarray, beam_row: pd.Series) -> List[Dict[str, float]]:
        """Convert 1D peak positions to 3D coordinates"""
        
        if len(peaks) == 0:
            return []
        
        beam_start = np.array([beam_row["Start X"], beam_row["Start Y"], beam_row["Start Z"]])
        beam_end = np.array([beam_row["End X"], beam_row["End Y"], beam_row["End Z"]])
        beam_length = self._calc_beam_length(beam_row)
        
        coordinates = []
        for i, peak_idx in enumerate(peaks):
            # Convert segment index to relative position
            rel_pos = peak_idx * self.segment_size / beam_length
            rel_pos = np.clip(rel_pos, 0.0, 1.0)
            
            # Interpolate 3D position
            coord_3d = beam_start + rel_pos * (beam_end - beam_start)
            
            coordinates.append({
                "column_id": f"col_{i+1}",
                "x": float(coord_3d[0]),
                "y": float(coord_3d[1]),
                "z": float(coord_3d[2])
            })
        
        return coordinates
    
    async def predict(self, building_data: Dict[str, Any], 
                     stage2_constraints: Dict[str, int]) -> Dict[str, Any]:
        """Run Stage 3 prediction with Stage 2 constraints"""
        
        if not self.loaded:
            await self.load_model()
        
        start_time = time.time()
        
        try:
            # Load data
            if "file_paths" in building_data:
                feature_df = pd.read_csv(building_data["file_paths"]["feature_matrix"])
                beam_wall_df = pd.read_csv(building_data["file_paths"]["beam_wall_matrix"], index_col=0)
                beam_beam_df = pd.read_csv(building_data["file_paths"]["beam_beam_matrix"], index_col=0)
            else:
                feature_df = pd.DataFrame(building_data["feature_matrix"])
                beam_wall_df = pd.DataFrame(building_data["beam_wall_matrix"])
                beam_beam_df = pd.DataFrame(building_data["beam_beam_matrix"])
            
            all_coordinates = []
            constraints_applied = 0
            constraints_satisfied = 0
            
            # Process each beam
            for beam_id, max_columns in stage2_constraints.items():
                try:
                    # Generate signals
                    wall_signal, beam_signal = self._generate_beam_signals(
                        feature_df, beam_wall_df, beam_beam_df, beam_id
                    )
                    
                    # Prepare input for CNN
                    input_signal = np.stack([wall_signal, beam_signal])
                    input_tensor = torch.tensor(input_signal, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Run CNN inference
                    with torch.no_grad():
                        output_signal = self.model(input_tensor)
                        output_signal = output_signal.cpu().numpy().squeeze()
                    
                    # Apply constraints and extract peaks
                    constrained_peaks = self._extract_constrained_peaks(output_signal, max_columns)
                    constraints_applied += 1
                    
                    if len(constrained_peaks) == max_columns:
                        constraints_satisfied += 1
                    
                    # Convert to coordinates
                    beam_info = feature_df[
                        (feature_df["Element Type"] == "Structural Framing") &
                        (feature_df["Element ID"].astype(str).str.contains(beam_id))
                    ]
                    
                    if not beam_info.empty:
                        beam_row = beam_info.iloc[0]
                        coordinates = self._convert_peaks_to_coordinates(constrained_peaks, beam_row)
                        
                        for coord in coordinates:
                            all_coordinates.append({
                                "building_id": building_data["building_id"],
                                "beam_id": beam_id,
                                "column_id": coord["column_id"],
                                "x": coord["x"],
                                "y": coord["y"],
                                "z": coord["z"],
                                "confidence": None  # Could add confidence from CNN output
                            })
                
                except Exception as e:
                    logger.warning(f"Failed to process beam {beam_id}: {str(e)}")
                    continue
            
            processing_time = time.time() - start_time
            
            # Create summary
            constraint_summary = {
                "total_coordinates": len(all_coordinates),
                "constraints_applied": constraints_applied,
                "constraints_satisfied": constraints_satisfied,
                "processing_time": processing_time
            }
            
            return {
                "coordinates": all_coordinates,
                "constraint_summary": constraint_summary,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Stage 3 prediction failed: {str(e)}")
            raise