#!/usr/bin/env python3
"""
Stage 2 Model: Column Count Prediction
Production wrapper for Stage 2 Graph Neural Network
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
import os
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
import time

logger = logging.getLogger(__name__)

class Stage2GNN(torch.nn.Module):
    """Graph Neural Network for Stage 2 column count prediction - matches training pipeline"""
    
    def __init__(self, input_dim: int = 15, hidden_dim: int = 32, num_classes: int = 3):
        super().__init__()
        self.c1 = SAGEConv(input_dim, hidden_dim)
        self.c2 = SAGEConv(hidden_dim, hidden_dim)
        self.head        = torch.nn.Linear(hidden_dim, num_classes)  # column count: 0/1/2
        self.mat_head    = torch.nn.Linear(hidden_dim, 8)            # column structural material
        self.length_head = torch.nn.Linear(hidden_dim, 1)            # avg column length (regression)
        
    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, 0.35, self.training)
        x = F.relu(self.c2(x, edge_index))
        return self.head(x), self.mat_head(x), self.length_head(x)

class Stage2Predictor:
    """Production wrapper for Stage 2 model"""
    
    def __init__(self, model_path: str = "models/stage2_model.pth"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.material_mapping = self._get_material_mapping()
        self.family_mapping = {}
        self.loaded = False
        
    def _get_material_mapping(self) -> Dict[str, int]:
        """Material mapping for feature encoding - matches training pipeline"""
        return {
            "Metal - Steel 43-275": 0,
            "Metal - Scaffold Tube": 1,
            "Steel 50-355": 2,
            "Steel 43-275": 3,
            "Stainless Steel": 4,
            "Metal - Steel 43-275 - existing": 5,
            "Steel 43-275 EXISTING": 6,
            "Unknown": 7
        }
    
    async def load_model(self):
        """Load the trained Stage 2 model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
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
                # Checkpoint contains metadata, extract model state dict and input dimension
                state_dict = checkpoint["model_state_dict"]
                input_dim = checkpoint.get("in_dim", 15)  # Default to 15 if not found
                logger.info(f"Loading model from checkpoint with metadata, input_dim: {input_dim}")
            else:
                # Direct state dict - try to infer input dimension from first layer
                state_dict = checkpoint
                # Get input dimension from first layer weight shape
                first_layer_key = "c1.lin_l.weight"
                if first_layer_key in state_dict:
                    input_dim = state_dict[first_layer_key].shape[1]
                else:
                    input_dim = 15  # Default fallback
                logger.info(f"Loading model from direct state dict, inferred input_dim: {input_dim}")
            
            # Initialize model with correct input dimension
            self.model = Stage2GNN(input_dim=input_dim)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            logger.info(f"Stage 2 model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load Stage 2 model: {str(e)}")
            raise
    
    def _norm_id(self, x) -> str:
        """Normalize element ID"""
        x = str(x).strip().replace(".0", "")
        nums = re.findall(r'\d+', x)
        
        if len(nums) >= 2:
            return nums[1]
        if len(nums) == 1:
            return nums[0]
        return x
    
    
    def _extract_features(self, feature_df: pd.DataFrame) -> Tuple[torch.Tensor, List[str], List[str], List[str], np.ndarray]:
        """Extract node features from FeatureMatrix - matches updated training pipeline"""
        
        # Clean data
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Include columns in the graph (changed from training pipeline)
        wall_df = feature_df[feature_df["Element Type"].str.contains("Wall", case=False, na=False)]
        beam_df = feature_df[feature_df["Element Type"].str.contains("Beam|Framing", case=False, na=False)]
        column_df = feature_df[feature_df["Element Type"].str.contains("Column", case=False, na=False)]
        
        if len(beam_df) == 0:
            raise ValueError("No beams found in FeatureMatrix")
        
        # Define features as used in training
        WALL_FEATS = ["Start X", "Start Y", "Start Z", "End X", "End Y", "End Z", "Length", "Area"]
        BEAM_FEATS = ["Start X", "Start Y", "Start Z", "End X", "End Y", "End Z",
                      "Length", "Volume", "Entity Start Level", "Entity End Level"]
        COLUMN_FEATS = ["Start X", "Start Y", "Start Z", "End X", "End Y", "End Z",
                        "Length", "Volume", "Entity Start Level", "Entity End Level"]
        
        # Extract features for each element type
        wall_x = torch.tensor(wall_df[WALL_FEATS].values, dtype=torch.float) if len(wall_df) > 0 else torch.empty(0, len(WALL_FEATS))
        beam_x = torch.tensor(beam_df[BEAM_FEATS].values, dtype=torch.float) if len(beam_df) > 0 else torch.empty(0, len(BEAM_FEATS))
        column_x = torch.tensor(column_df[COLUMN_FEATS].values, dtype=torch.float) if len(column_df) > 0 else torch.empty(0, len(COLUMN_FEATS))
        
        # Add normalized beam length
        if len(beam_df) > 0:
            bl = beam_df["Length"].values
            bl_norm = bl / (np.mean(bl) + 1e-6)
            bl_feat = torch.tensor(bl_norm, dtype=torch.float).unsqueeze(1)
            beam_x = torch.cat([beam_x, bl_feat], dim=1)
        
        # Add normalized column length
        if len(column_df) > 0:
            cl = column_df["Length"].values
            cl_norm = cl / (np.mean(cl) + 1e-6)
            cl_feat = torch.tensor(cl_norm, dtype=torch.float).unsqueeze(1)
            column_x = torch.cat([column_x, cl_feat], dim=1)

        # Structural Material as encoded integer feature (column only)
        # Always account for material feature in max_dim calculation
        if len(column_df) > 0:
            material_column = "Structural Material" if "Structural Material" in column_df.columns else "Material"
            if material_column not in column_df.columns:
                column_df = column_df.copy()
                column_df[material_column] = "Unknown"
            
            mat_ids = column_df[material_column].apply(lambda x: self.material_mapping.get(str(x).strip(), self.material_mapping["Unknown"])).values
            mat_feat = torch.tensor(mat_ids, dtype=torch.float).unsqueeze(1)
            column_x = torch.cat([column_x, mat_feat], dim=1)
        else:
            mat_ids = np.array([])
            # Create empty column tensor for consistency
            column_x = torch.empty(0, len(COLUMN_FEATS))

        # Pad dimensions - ensure consistent max_dim across all buildings
        max_dim = 0
        if len(wall_df) > 0:
            max_dim = max(max_dim, wall_x.shape[1])
        if len(beam_df) > 0:
            max_dim = max(max_dim, beam_x.shape[1])
        if len(column_df) > 0:
            max_dim = max(max_dim, column_x.shape[1])
        else:
            # When no columns, account for the material feature that would have been added
            # Base column features (10) + normalized length (1) + material (1) = 12
            expected_column_dim = 12
            max_dim = max(max_dim, expected_column_dim)
        
        # Pad all feature matrices to max_dim
        if len(wall_df) > 0:
            wall_x = F.pad(wall_x, (0, max_dim - wall_x.shape[1]))
        if len(beam_df) > 0:
            beam_x = F.pad(beam_x, (0, max_dim - beam_x.shape[1]))
        # Only pad column_x if there are columns
        if len(column_df) > 0:
            column_x = F.pad(column_x, (0, max_dim - column_x.shape[1]))
        
        # Type flags: 2 for walls, 1 for beams, 0 for columns
        if len(wall_df) > 0:
            wall_x = torch.cat([wall_x, torch.full((wall_x.size(0), 1), 2)], dim=1)
        if len(beam_df) > 0:
            beam_x = torch.cat([beam_x, torch.ones((beam_x.size(0), 1))], dim=1)
        if len(column_df) > 0:
            column_x = torch.cat([column_x, torch.zeros((column_x.size(0), 1))], dim=1)
        
        # Combine all node types
        x_parts = []
        if len(wall_df) > 0:
            x_parts.append(wall_x)
        if len(beam_df) > 0:
            x_parts.append(beam_x)
        if len(column_df) > 0:
            x_parts.append(column_x)
        
        x = torch.cat(x_parts, dim=0) if x_parts else torch.empty(0, max_dim + 1)
        
        # Extract IDs
        wall_ids = wall_df["Element ID"].astype(str).apply(self._norm_id).tolist() if len(wall_df) > 0 else []
        beam_ids = beam_df["Element ID"].astype(str).apply(self._norm_id).tolist() if len(beam_df) > 0 else []
        column_ids = column_df["Element ID"].astype(str).apply(self._norm_id).tolist() if len(column_df) > 0 else []
        
        return x, wall_ids, beam_ids, column_ids, mat_ids
    
    def _build_graph(self, feature_df: pd.DataFrame, beam_wall_df: pd.DataFrame, 
                    beam_column_df: pd.DataFrame, beam_beam_df: pd.DataFrame = None) -> Tuple[Data, List[str], List[str], np.ndarray]:
        """Build graph data structure - matches updated training pipeline"""
        
        # Extract features (now includes walls, beams, and columns)
        x, wall_ids, beam_ids, column_ids, mat_ids = self._extract_features(feature_df)
        
        # Create node ID mapping (walls + beams + columns - matches training)
        node_ids = wall_ids + beam_ids + column_ids
        id_to_idx = {eid: i for i, eid in enumerate(node_ids)}
        
        # Build edges and support statistics
        edges = []
        support_degree = torch.zeros(len(node_ids))
        endpoint_support = torch.zeros(len(node_ids))
        
        # Add beam-wall edges
        if not beam_wall_df.empty:
            for _, row in beam_wall_df.iterrows():
                b = self._norm_id(row.iloc[0])
                if b not in id_to_idx:
                    continue
                
                bi = id_to_idx[b]
                connected = 0
                
                for w_raw, val in row.iloc[1:].items():
                    if val == 1:
                        w = self._norm_id(w_raw)
                        if w in id_to_idx:
                            wi = id_to_idx[w]
                            edges.append([bi, wi])
                            edges.append([wi, bi])
                            connected += 1
                
                support_degree[bi] = connected
                endpoint_support[bi] = 1 if connected >= 2 else 0
        
        # Add beam-column edges - SKIP FOR PREDICTION MODE
        # We want to predict required columns WITHOUT seeing existing column connections
        # if not beam_column_df.empty:
        #     for _, row in beam_column_df.iterrows():
        #         b = self._norm_id(row.iloc[0])
        #         if b not in id_to_idx:
        #             continue
        #         
        #         bi = id_to_idx[b]
        #         
        #         for c_raw, val in row.iloc[1:].items():
        #             if val == 1:
        #                 c = self._norm_id(c_raw)
        #                 if c in id_to_idx:
        #                     ci = id_to_idx[c]
        #                     edges.append([bi, ci])
        #                     edges.append([ci, bi])
        
        logger.info("Prediction mode: Skipping beam-column edges to predict required columns")
        
        # Add beam-beam edges
        if beam_beam_df is not None and not beam_beam_df.empty:
            for _, row in beam_beam_df.iterrows():
                b1 = self._norm_id(row.iloc[0])
                if b1 not in id_to_idx:
                    continue
                
                b1i = id_to_idx[b1]
                
                for b2_raw, val in row.iloc[1:].items():
                    if val == 1:
                        b2 = self._norm_id(b2_raw)
                        if b2 in id_to_idx:
                            b2i = id_to_idx[b2]
                            edges.append([b1i, b2i])
                            edges.append([b2i, b1i])
        
        # Create edge index
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).T
        else:
            # Create self-loops if no edges
            edge_index = torch.arange(x.size(0)).repeat(2, 1)
        
        # Append support features (matches training pipeline)
        x = torch.cat([
            x,
            support_degree.unsqueeze(1),
            endpoint_support.unsqueeze(1)
        ], dim=1)
        
        return Data(x=x, edge_index=edge_index), beam_ids, column_ids, mat_ids
    
    async def predict(self, building_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage 2 prediction"""
        
        if not self.loaded:
            await self.load_model()
        
        start_time = time.time()
        
        try:
            logger.info("Stage 2 prediction started")
            
            # Load data
            if "file_paths" in building_data:
                # Load from files
                feature_df = pd.read_csv(building_data["file_paths"]["feature_matrix"])
                beam_wall_df = pd.read_csv(building_data["file_paths"]["beam_wall_matrix"], index_col=0)
                beam_column_df = pd.read_csv(building_data["file_paths"]["beam_column_matrix"], index_col=0)
                beam_beam_df = pd.read_csv(building_data["file_paths"]["beam_beam_matrix"], index_col=0) if "beam_beam_matrix" in building_data["file_paths"] else pd.DataFrame()
                logger.info(f"Loaded data from files - Features: {len(feature_df)}, BW: {beam_wall_df.shape}, BC: {beam_column_df.shape}")
            else:
                # Load from data
                feature_df = pd.DataFrame(building_data["feature_matrix"])
                beam_wall_df = pd.DataFrame(building_data["beam_wall_matrix"])
                beam_column_df = pd.DataFrame(building_data["beam_column_matrix"])
                beam_beam_df = pd.DataFrame(building_data.get("beam_beam_matrix", []))
                logger.info(f"Loaded data from memory - Features: {len(feature_df)}")
            
            # Build graph
            logger.info("Building graph...")
            
            # Debug: Log beam-column connections from input data
            logger.info(f"BeamColumnMatrix shape: {beam_column_df.shape}")
            if not beam_column_df.empty:
                logger.info(f"BeamColumnMatrix sample:\n{beam_column_df.head()}")
            else:
                logger.info("BeamColumnMatrix is empty - no existing columns")
            
            graph_data, beam_ids, column_ids, mat_ids = self._build_graph(feature_df, beam_wall_df, beam_column_df, beam_beam_df)
            logger.info(f"Graph built - Nodes: {graph_data.x.shape[0]}, Features: {graph_data.x.shape[1]}, Beams: {len(beam_ids)}, Columns: {len(column_ids)}")
            
            # Debug: Check feature statistics
            logger.info(f"Feature tensor stats - Min: {graph_data.x.min().item():.3f}, Max: {graph_data.x.max().item():.3f}, Mean: {graph_data.x.mean().item():.3f}")
            logger.info(f"Edge connections: {graph_data.edge_index.shape[1]} edges")
            
            graph_data = graph_data.to(self.device)
            
            # Run inference
            logger.info("Running inference...")
            with torch.no_grad():
                column_pred, material_pred, length_pred = self.model(graph_data.x, graph_data.edge_index)
                column_probs = F.softmax(column_pred, dim=1)
                column_predictions = torch.argmax(column_probs, dim=1)
                material_predictions = torch.argmax(material_pred, dim=1)
                
                # Debug: Log prediction statistics
                unique_preds, counts = torch.unique(column_predictions, return_counts=True)
                logger.info(f"Prediction distribution: {dict(zip(unique_preds.tolist(), counts.tolist()))}")
                logger.info(f"Sample predictions (first 5): {column_predictions[:5].tolist()}")
                logger.info(f"Sample probabilities (first 5): {column_probs[:5].tolist()}")
            
            logger.info("Inference completed, formatting results...")
            
            # Extract predictions for beams only
            # In the graph: walls + beams + columns
            wall_df = feature_df[feature_df["Element Type"].str.contains("Wall", case=False, na=False)]
            num_walls = len(wall_df)
            num_beams = len(beam_ids)
            
            logger.info(f"Node distribution - Walls: {num_walls}, Beams: {num_beams}, Columns: {len(column_ids)}")
            logger.info(f"Beam indices in graph: {num_walls} to {num_walls + num_beams - 1}")
            
            # Format results for beams only (column count prediction)
            beam_predictions = []
            predictions_by_count = {"0": 0, "1": 0, "2": 0}
            
            for i, beam_id in enumerate(beam_ids):
                graph_idx = num_walls + i  # Beams start after walls in the graph
                pred_columns = int(column_predictions[graph_idx].item())
                confidence = float(column_probs[graph_idx, pred_columns].item())
                
                # Predict material and length for beams that need columns
                predicted_material = None
                material_confidence = None
                predicted_column_length = None
                if pred_columns > 0:
                    # Material prediction
                    pred_material_id = int(material_predictions[graph_idx].item())
                    material_probs_beam = F.softmax(material_pred[graph_idx], dim=0)
                    material_confidence = float(material_probs_beam[pred_material_id].item())
                    material_names = {v: k for k, v in self.material_mapping.items()}
                    predicted_material = material_names.get(pred_material_id, "Unknown")
                    
                    # Column length prediction
                    predicted_column_length = float(length_pred[graph_idx].item())
                
                logger.info(f"Beam {beam_id} (graph_idx {graph_idx}): {pred_columns} columns, confidence: {confidence:.3f}, material: {predicted_material}, length: {predicted_column_length}")
                
                beam_predictions.append({
                    "beam_id": beam_id,
                    "predicted_columns": pred_columns,
                    "confidence": confidence,
                    "predicted_material": predicted_material,
                    "material_confidence": material_confidence,
                    "predicted_column_length": predicted_column_length,
                    "type": "beam"
                })
                
                predictions_by_count[str(pred_columns)] += 1
            
            # Format results for columns (material prediction)
            column_predictions_list = []
            if len(column_ids) > 0:
                for i, column_id in enumerate(column_ids):
                    graph_idx = num_walls + num_beams + i  # Columns start after walls and beams
                    pred_material = int(material_predictions[graph_idx].item())
                    material_probs = F.softmax(material_pred[graph_idx], dim=0)
                    material_confidence = float(material_probs[pred_material].item())
                    
                    # Get material name
                    material_names = {v: k for k, v in self.material_mapping.items()}
                    material_name = material_names.get(pred_material, "Unknown")
                    
                    column_predictions_list.append({
                        "column_id": column_id,
                        "predicted_material": material_name,
                        "material_confidence": material_confidence,
                        "material_id": pred_material,
                        "type": "column"  # Add type field for consistency
                    })
            
            processing_time = time.time() - start_time
            
            # Create summary
            lengths = [p["predicted_column_length"] for p in beam_predictions if p["predicted_column_length"] is not None]
            summary = {
                "total_beams": len(beam_ids),
                "total_columns": len(column_ids),
                "predictions_by_count": predictions_by_count,
                "average_confidence": float(np.mean([p["confidence"] for p in beam_predictions])) if beam_predictions else 0.0,
                "average_material_confidence": float(np.mean([p["material_confidence"] for p in column_predictions_list])) if column_predictions_list else 0.0,
                "average_predicted_column_length": float(np.mean(lengths)) if lengths else 0.0,
                "processing_time": processing_time
            }
            
            logger.info(f"Stage 2 prediction completed successfully - {len(beam_predictions)} beam predictions, {len(column_predictions_list)} column predictions")
            
            return {
                "predictions": beam_predictions,  # Keep original key for compatibility
                "beam_predictions": beam_predictions,  # New detailed structure
                "column_predictions": column_predictions_list,
                "summary": summary,
                "processing_time": processing_time,
                # Add individual prediction lists with type field for consistency with validation script
                "results": beam_predictions + [
                    {**col_pred, "type": "column"} for col_pred in column_predictions_list
                ]
            }
            
        except Exception as e:
            logger.error(f"Stage 2 prediction failed: {str(e)}", exc_info=True)
            raise