#!/usr/bin/env python3
"""
CSV Processor: Convert single CSV to connection matrices
Based on the provided connection detection logic
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Tuple, Optional, Any
import tempfile

logger = logging.getLogger(__name__)

class CSVProcessor:
    """Process single CSV file to generate connection matrices"""
    
    def __init__(self):
        # Parameters (in feet)
        self.DEFAULT_BEAM_WIDTH = 0.7
        self.DEFAULT_WALL_WIDTH = 0.7
        self.COLUMN_RADIUS = 0.4
        self.INTERSECTION_TOL = 1
        self.VERTICAL_BEAM_TOL = 0.4
        self.VERTICAL_COLUMN_TOL = 3
        self.VERTICAL_WALL_TOL = 0.3
        self.COLUMN_START_TOL = 0.2
    
    def point_to_segment_distance(self, px: float, py: float, 
                                 x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate distance from point to line segment"""
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return np.hypot(px - x1, py - y1)
        
        t = ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t))
        
        proj_x = x1 + t*dx
        proj_y = y1 + t*dy
        
        return np.hypot(px - proj_x, py - proj_y)
    
    def z_overlap(self, min1: float, max1: float, min2: float, max2: float, tol: float) -> bool:
        """Check if two Z ranges overlap within tolerance"""
        return not (max1 < min2 - tol or min1 > max2 + tol)
    
    def filter_elements(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter and separate beams, columns, and walls"""
        
        # Filter beams
        beams = df[
            (df["Element Type"] == "Structural Framing") &
            (~df["Family"].str.contains("Lintel", case=False, na=False)) &
            (df["Structural Material"].str.contains("Metal|Steel", case=False, na=False))
        ].copy()
        
        # Filter columns
        columns = df[df["Element Type"] == "Structural Column"].copy()
        
        # Filter walls
        walls = df[
            (df["Element Type"] == "Wall") &
            (df["Width"] > 0.5)
        ].copy()
        
        return beams, columns, walls
    
    def setup_element_properties(self, beams: pd.DataFrame, columns: pd.DataFrame, 
                                walls: pd.DataFrame, node_name: str) -> Tuple[Dict, Dict, Dict]:
        """Setup element properties and ID mappings"""
        
        # Apply width defaults
        beams["Width"] = beams.get("Width", self.DEFAULT_BEAM_WIDTH).fillna(self.DEFAULT_BEAM_WIDTH)
        walls["Width"] = walls.get("Width", self.DEFAULT_WALL_WIDTH).fillna(self.DEFAULT_WALL_WIDTH)
        
        # Calculate Z ranges
        beams["Zmin"] = beams[["Start Z", "End Z"]].min(axis=1)
        beams["Zmax"] = beams[["Start Z", "End Z"]].max(axis=1)
        
        columns["Zmin"] = columns[["Start Z", "End Z"]].min(axis=1)
        columns["Zmax"] = columns[["Start Z", "End Z"]].max(axis=1)
        
        walls["Zmin"] = walls[["Entity Start Level", "Entity End Level"]].min(axis=1)
        walls["Zmax"] = walls[["Entity Start Level", "Entity End Level"]].max(axis=1)
        
        # Create ID mappings
        beam_ids = {b["Element ID"]: f"{node_name}_{b['Element ID']}_B"
                   for _, b in beams.iterrows()}
        
        col_ids = {c["Element ID"]: f"{node_name}_{c['Element ID']}_C"
                  for _, c in columns.iterrows()}
        
        wall_ids = {w["Element ID"]: f"{node_name}_{w['Element ID']}_W"
                   for _, w in walls.iterrows()}
        
        return beam_ids, col_ids, wall_ids
    
    def detect_connections(self, beams: pd.DataFrame, columns: pd.DataFrame, 
                          walls: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
        """Detect beam connections using geometric analysis"""
        
        # Initialize connection storage
        beam_beam_links = {bid: set() for bid in beams["Element ID"]}
        beam_col_links = {bid: set() for bid in beams["Element ID"]}
        beam_wall_links = {bid: set() for bid in beams["Element ID"]}
        
        # Process each beam
        for _, beam in beams.iterrows():
            bid = beam["Element ID"]
            beam_half = beam["Width"] / 2
            
            x1, y1, z1 = beam["Start X"], beam["Start Y"], beam["Start Z"]
            x2, y2, z2 = beam["End X"], beam["End Y"], beam["End Z"]
            
            beam_min_z = beam["Zmin"]
            beam_max_z = beam["Zmax"]
            
            endpoints = [(x1, y1, z1), (x2, y2, z2)]
            
            for px, py, pz in endpoints:
                column_found = False
                
                # BEAM → BEAM connections
                for _, other in beams.iterrows():
                    oid = other["Element ID"]
                    if oid == bid:
                        continue
                    
                    if not self.z_overlap(
                        beam_min_z, beam_max_z,
                        other["Zmin"], other["Zmax"],
                        self.VERTICAL_BEAM_TOL):
                        continue
                    
                    other_half = other["Width"] / 2
                    
                    d = self.point_to_segment_distance(
                        px, py,
                        other["Start X"], other["Start Y"],
                        other["End X"], other["End Y"]
                    )
                    
                    if d <= (beam_half + other_half + self.INTERSECTION_TOL):
                        beam_beam_links[bid].add(oid)
                
                # BEAM → COLUMN connections
                for _, col in columns.iterrows():
                    col_bot = col["Zmin"]
                    col_top = col["Zmax"]
                    col_mid = (col_bot + col_top) / 2
                    
                    if not self.z_overlap(
                        beam_min_z, beam_max_z,
                        col_bot, col_top,
                        self.VERTICAL_COLUMN_TOL):
                        continue
                    
                    # Skip column starting from beam
                    if abs(pz - col_bot) < self.COLUMN_START_TOL:
                        continue
                    
                    # Beam must hit upper half
                    if pz < col_mid:
                        continue
                    
                    col_x = col["Start X"]
                    col_y = col["Start Y"]
                    
                    d = np.hypot(px - col_x, py - col_y)
                    
                    if d <= (beam_half + self.COLUMN_RADIUS + self.INTERSECTION_TOL):
                        beam_col_links[bid].add(col["Element ID"])
                        column_found = True
                        break
                
                # BEAM → WALL connections (only if no column found)
                if column_found:
                    continue
                
                for _, wall in walls.iterrows():
                    if not self.z_overlap(
                        beam_min_z, beam_max_z,
                        wall["Zmin"], wall["Zmax"],
                        self.VERTICAL_WALL_TOL):
                        continue
                    
                    wall_half = wall["Width"] / 2
                    
                    d = self.point_to_segment_distance(
                        px, py,
                        wall["Start X"], wall["Start Y"],
                        wall["End X"], wall["End Y"]
                    )
                    
                    if d <= (wall_half + self.INTERSECTION_TOL):
                        beam_wall_links[bid].add(wall["Element ID"])
        
        return beam_beam_links, beam_col_links, beam_wall_links
    
    def create_matrices(self, beam_beam_links: Dict, beam_col_links: Dict, beam_wall_links: Dict,
                       beam_ids: Dict, col_ids: Dict, wall_ids: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create connection matrices"""
        
        # Beam-Beam Matrix
        bb_matrix = pd.DataFrame(
            0,
            index=[beam_ids[b] for b in beam_beam_links],
            columns=[beam_ids[b] for b in beam_beam_links]
        )
        
        # Beam-Column Matrix (handle empty columns case)
        if col_ids:
            bc_matrix = pd.DataFrame(
                0,
                index=[beam_ids[b] for b in beam_col_links],
                columns=[col_ids[c] for c in col_ids]
            )
        else:
            bc_matrix = pd.DataFrame(
                0,
                index=[beam_ids[b] for b in beam_col_links],
                columns=[]
            )
        
        # Beam-Wall Matrix
        bw_matrix = pd.DataFrame(
            0,
            index=[beam_ids[b] for b in beam_wall_links],
            columns=[wall_ids[w] for w in wall_ids]
        )
        
        # Fill matrices with connections
        for b, others in beam_beam_links.items():
            for o in others:
                if o in beam_ids:  # Ensure other beam exists
                    bb_matrix.loc[beam_ids[b], beam_ids[o]] = 1
        
        for b, cols in beam_col_links.items():
            for c in cols:
                if c in col_ids:  # Ensure column exists
                    bc_matrix.loc[beam_ids[b], col_ids[c]] = 1
        
        for b, ws in beam_wall_links.items():
            for w in ws:
                if w in wall_ids:  # Ensure wall exists
                    bw_matrix.loc[beam_ids[b], wall_ids[w]] = 1
        
        return bb_matrix, bc_matrix, bw_matrix
    
    def process_csv(self, csv_content: str, building_id: str) -> Dict[str, Any]:
        """
        Process single CSV file and generate all required matrices
        
        Args:
            csv_content: CSV file content as string
            building_id: Building identifier
            
        Returns:
            Dictionary containing all matrices and feature data
        """
        
        try:
            # Parse CSV content
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content))
            
            logger.info(f"Processing CSV for building {building_id}")
            logger.info(f"Total elements in CSV: {len(df)}")
            
            # Extract node name from building_id or use building_id
            node_name = building_id
            
            # Filter elements
            beams, columns, walls = self.filter_elements(df)
            
            logger.info(f"Filtered elements - Beams: {len(beams)}, Columns: {len(columns)}, Walls: {len(walls)}")
            
            if len(beams) == 0:
                raise ValueError("No structural framing elements found in CSV")
            
            # Setup properties and ID mappings
            beam_ids, col_ids, wall_ids = self.setup_element_properties(beams, columns, walls, node_name)
            
            # Detect connections
            beam_beam_links, beam_col_links, beam_wall_links = self.detect_connections(beams, columns, walls)
            
            # Create matrices
            bb_matrix, bc_matrix, bw_matrix = self.create_matrices(
                beam_beam_links, beam_col_links, beam_wall_links,
                beam_ids, col_ids, wall_ids
            )
            
            # Create feature matrix
            feature_df = pd.concat([beams, columns, walls], ignore_index=True)
            feature_df.insert(0, "Node Name", node_name)
            
            # Save matrices to temporary files
            temp_dir = tempfile.mkdtemp()
            
            file_paths = {}
            
            # Save BeamBeamMatrix
            bb_path = os.path.join(temp_dir, f"{building_id}_BeamBeamMatrix.csv")
            bb_matrix.to_csv(bb_path)
            file_paths["beam_beam_matrix"] = bb_path
            
            # Save BeamColumnMatrix (even if empty)
            bc_path = os.path.join(temp_dir, f"{building_id}_BeamColumnMatrix.csv")
            bc_matrix.to_csv(bc_path)
            file_paths["beam_column_matrix"] = bc_path
            
            # Save BeamWallMatrix
            bw_path = os.path.join(temp_dir, f"{building_id}_BeamWallMatrix.csv")
            bw_matrix.to_csv(bw_path)
            file_paths["beam_wall_matrix"] = bw_path
            
            # Save FeatureMatrix
            feature_path = os.path.join(temp_dir, f"{building_id}_FeatureMatrix.csv")
            feature_df.to_csv(feature_path, index=False)
            file_paths["feature_matrix"] = feature_path
            
            # Create connection summary
            connection_summary = {
                "total_beams": len(beams),
                "total_columns": len(columns),
                "total_walls": len(walls),
                "beam_beam_connections": sum(len(links) for links in beam_beam_links.values()),
                "beam_column_connections": sum(len(links) for links in beam_col_links.values()),
                "beam_wall_connections": sum(len(links) for links in beam_wall_links.values())
            }
            
            logger.info(f"Connection detection completed: {connection_summary}")
            
            return {
                "building_id": building_id,
                "file_paths": file_paths,
                "matrices": {
                    "beam_beam_matrix": bb_matrix.to_dict(),
                    "beam_column_matrix": bc_matrix.to_dict(),
                    "beam_wall_matrix": bw_matrix.to_dict()
                },
                "feature_matrix": feature_df.to_dict('records'),
                "connection_summary": connection_summary,
                "temp_dir": temp_dir
            }
            
        except Exception as e:
            logger.error(f"CSV processing failed: {str(e)}")
            raise