#!/usr/bin/env python3
"""
Test CSV Processing Functionality
Test the new CSV processing endpoints that generate connection matrices
"""

import requests
import pandas as pd
import numpy as np
import os
import tempfile
from typing import Dict, Any

class CSVProcessingTester:
    """Test client for CSV processing functionality"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def create_sample_csv(self) -> str:
        """Create a comprehensive sample CSV with beams, columns, and walls"""
        
        # Sample building data with realistic coordinates
        data = [
            # Beams (Structural Framing)
            {
                "Element ID": "beam_001",
                "Element Type": "Structural Framing",
                "Family": "W12x26",
                "Structural Material": "Steel 43-275",
                "Start X": 0, "Start Y": 0, "Start Z": 10,
                "End X": 20, "End Y": 0, "End Z": 10,
                "Width": 0.5, "Height": 1.0
            },
            {
                "Element ID": "beam_002", 
                "Element Type": "Structural Framing",
                "Family": "W12x26",
                "Structural Material": "Steel 43-275",
                "Start X": 20, "Start Y": 0, "Start Z": 10,
                "End X": 40, "End Y": 0, "End Z": 10,
                "Width": 0.5, "Height": 1.0
            },
            {
                "Element ID": "beam_003",
                "Element Type": "Structural Framing", 
                "Family": "W12x26",
                "Structural Material": "Steel 43-275",
                "Start X": 0, "Start Y": 10, "Start Z": 10,
                "End X": 20, "End Y": 10, "End Z": 10,
                "Width": 0.5, "Height": 1.0
            },
            {
                "Element ID": "beam_004",
                "Element Type": "Structural Framing",
                "Family": "W12x26", 
                "Structural Material": "Steel 43-275",
                "Start X": 20, "Start Y": 10, "Start Z": 10,
                "End X": 40, "End Y": 10, "End Z": 10,
                "Width": 0.5, "Height": 1.0
            },
            # Columns
            {
                "Element ID": "col_001",
                "Element Type": "Structural Column",
                "Family": "HSS8x8x1/2",
                "Structural Material": "Steel 43-275",
                "Start X": 20, "Start Y": 0, "Start Z": 0,
                "End X": 20, "End Y": 0, "End Z": 15,
                "Width": 0.8, "Height": 0.8
            },
            {
                "Element ID": "col_002",
                "Element Type": "Structural Column",
                "Family": "HSS8x8x1/2",
                "Structural Material": "Steel 43-275", 
                "Start X": 20, "Start Y": 10, "Start Z": 0,
                "End X": 20, "End Y": 10, "End Z": 15,
                "Width": 0.8, "Height": 0.8
            },
            # Walls
            {
                "Element ID": "wall_001",
                "Element Type": "Wall",
                "Family": "Generic - 8\"",
                "Structural Material": "Concrete",
                "Start X": 0, "Start Y": -1, "Start Z": 0,
                "End X": 40, "End Y": -1, "End Z": 0,
                "Width": 0.67, "Height": 12,
                "Entity Start Level": 0, "Entity End Level": 12
            },
            {
                "Element ID": "wall_002",
                "Element Type": "Wall",
                "Family": "Generic - 8\"",
                "Structural Material": "Concrete",
                "Start X": 0, "Start Y": 11, "Start Z": 0,
                "End X": 40, "End Y": 11, "End Z": 0,
                "Width": 0.67, "Height": 12,
                "Entity Start Level": 0, "Entity End Level": 12
            }
        ]
        
        # Create DataFrame and save to temporary CSV
        df = pd.DataFrame(data)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name
    
    def test_csv_processing(self, building_id: str = "test_building_csv") -> Dict[str, Any]:
        """Test CSV processing endpoint"""
        
        # Create sample CSV
        csv_file_path = self.create_sample_csv()
        
        try:
            # Upload and process CSV
            with open(csv_file_path, 'rb') as f:
                files = {'csv_file': f}
                response = self.session.post(
                    f"{self.base_url}/process-csv?building_id={building_id}",
                    files=files
                )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ CSV processing successful")
                print(f"   Job ID: {data['job_id']}")
                print(f"   Building ID: {data['building_id']}")
                print(f"   Connection Summary:")
                for key, value in data['connection_summary'].items():
                    print(f"     {key}: {value}")
                print(f"   Generated files: {list(data['file_paths'].keys())}")
                return data
            else:
                print(f"❌ CSV processing failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return {}
                
        except Exception as e:
            print(f"❌ CSV processing error: {str(e)}")
            return {}
        finally:
            # Cleanup temporary file
            if os.path.exists(csv_file_path):
                os.remove(csv_file_path)
    
    def test_csv_with_stage2(self, building_id: str = "test_building_stage2") -> Dict[str, Any]:
        """Test CSV processing + Stage 2 inference endpoint"""
        
        # Create sample CSV
        csv_file_path = self.create_sample_csv()
        
        try:
            # Upload CSV and run Stage 2 inference
            with open(csv_file_path, 'rb') as f:
                files = {'csv_file': f}
                response = self.session.post(
                    f"{self.base_url}/process-csv-with-stage2?building_id={building_id}",
                    files=files
                )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ CSV processing + Stage 2 inference successful")
                print(f"   Job ID: {data['job_id']}")
                print(f"   Building ID: {data['building_id']}")
                print(f"   Connection Summary:")
                for key, value in data['connection_summary'].items():
                    print(f"     {key}: {value}")
                print(f"   Stage 2 Results:")
                print(f"     Total beams: {data['stage2_summary']['total_beams']}")
                print(f"     Average confidence: {data['stage2_summary']['average_confidence']:.3f}")
                print(f"     Predictions by count: {data['stage2_summary']['predictions_by_count']}")
                print(f"   Processing time: {data['processing_time']:.2f}s")
                
                # Show individual beam predictions
                print(f"   Individual Predictions:")
                for pred in data['stage2_predictions'][:5]:  # Show first 5
                    print(f"     {pred['beam_id']}: {pred['predicted_columns']} columns (conf: {pred['confidence']:.3f})")
                if len(data['stage2_predictions']) > 5:
                    print(f"     ... and {len(data['stage2_predictions']) - 5} more beams")
                
                return data
            else:
                print(f"❌ CSV processing + Stage 2 inference failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return {}
                
        except Exception as e:
            print(f"❌ CSV processing + Stage 2 inference error: {str(e)}")
            return {}
        finally:
            # Cleanup temporary file
            if os.path.exists(csv_file_path):
                os.remove(csv_file_path)
        """Test CSV processing + pipeline endpoint"""
        
        # Create sample CSV
        csv_file_path = self.create_sample_csv()
        
        try:
    def test_csv_and_pipeline(self, building_id: str = "test_building_pipeline") -> Dict[str, Any]:
        """Test CSV processing + full pipeline endpoint"""
        
        # Create sample CSV
        csv_file_path = self.create_sample_csv()
        
        try:
            # Upload CSV and run pipeline
            with open(csv_file_path, 'rb') as f:
                files = {'csv_file': f}
                response = self.session.post(
                    f"{self.base_url}/process-csv-and-predict?building_id={building_id}",
                    files=files
                )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ CSV processing + full pipeline successful")
                print(f"   Job ID: {data['job_id']}")
                print(f"   Stage 2 - Total beams: {data['stage2_results']['summary']['total_beams']}")
                print(f"   Stage 3 - Total coordinates: {data['stage3_results']['constraint_summary']['total_coordinates']}")
                print(f"   Total processing time: {data['total_processing_time']:.2f}s")
                return data
            else:
                print(f"❌ CSV processing + full pipeline failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return {}
                
        except Exception as e:
            print(f"❌ CSV processing + full pipeline error: {str(e)}")
            return {}
        finally:
            # Cleanup temporary file
            if os.path.exists(csv_file_path):
                os.remove(csv_file_path)
    
    def test_matrix_validation(self, csv_result: Dict[str, Any]) -> bool:
        """Validate generated matrices"""
        
        if not csv_result or "file_paths" not in csv_result:
            print("❌ No file paths to validate")
            return False
        
        file_paths = csv_result["file_paths"]
        required_files = ["beam_beam_matrix", "beam_column_matrix", "beam_wall_matrix", "feature_matrix"]
        
        print("\n📊 Validating generated matrices...")
        
        all_valid = True
        for file_type in required_files:
            if file_type not in file_paths:
                print(f"❌ Missing {file_type}")
                all_valid = False
                continue
            
            file_path = file_paths[file_type]
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                all_valid = False
                continue
            
            try:
                if file_type == "feature_matrix":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_csv(file_path, index_col=0)
                
                print(f"✅ {file_type}: {df.shape} - Valid")
                
                # Show sample data
                if file_type != "feature_matrix":
                    connections = (df == 1).sum().sum()
                    print(f"   Total connections: {connections}")
                
            except Exception as e:
                print(f"❌ Error reading {file_type}: {str(e)}")
                all_valid = False
        
        return all_valid
    
    def run_csv_tests(self):
        """Run all CSV processing tests"""
        
        print("=" * 70)
        print("CSV PROCESSING TESTS")
        print("=" * 70)
        
        # Test 1: CSV Processing Only
        print("\n1. Testing CSV processing endpoint...")
        csv_result = self.test_csv_processing()
        
        if csv_result:
            # Test 2: Matrix Validation
            print("\n2. Validating generated matrices...")
            self.test_matrix_validation(csv_result)
        
        # Test 3: CSV Processing + Stage 2 Inference
        print("\n3. Testing CSV processing + Stage 2 inference endpoint...")
        stage2_result = self.test_csv_with_stage2()
        
        # Test 4: CSV Processing + Full Pipeline
        print("\n4. Testing CSV processing + full pipeline endpoint...")
        pipeline_result = self.test_csv_and_pipeline()
        
        print("\n" + "=" * 70)
        print("CSV PROCESSING TESTS COMPLETED")
        print("=" * 70)

def main():
    """Main test function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CSV Processing API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", choices=["csv", "stage2", "pipeline", "all"], 
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    tester = CSVProcessingTester(args.url)
    
    if args.test == "all":
        tester.run_csv_tests()
    elif args.test == "csv":
        tester.test_csv_processing()
    elif args.test == "stage2":
        tester.test_csv_with_stage2()
    elif args.test == "pipeline":
        tester.test_csv_and_pipeline()

if __name__ == "__main__":
    main()