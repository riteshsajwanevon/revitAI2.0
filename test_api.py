#!/usr/bin/env python3
"""
API Testing Script
Test the Structural Analysis Pipeline API endpoints
"""

import requests
import json
import os
import time
from typing import Dict, Any

class APITester:
    """Test client for the Structural Analysis Pipeline API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self) -> bool:
        """Test health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check passed")
                print(f"   Status: {data['status']}")
                print(f"   Models: {data['models']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    def upload_test_files(self, building_id: str = "test_building") -> Dict[str, str]:
        """Upload test CSV files"""
        
        # Create sample CSV data
        feature_data = """Element ID,Element Type,Family,Material,Start X,Start Y,Start Z,End X,End Y,End Z,Width,Height
beam_001,Structural Framing,W12x26,Steel 43-275,0,0,0,10,0,0,0.3,0.3
beam_002,Structural Framing,W12x26,Steel 43-275,10,0,0,20,0,0,0.3,0.3
column_001,Structural Column,HSS8x8x1/2,Steel 43-275,5,0,0,5,0,3,0.2,0.2
wall_001,Wall,Generic - 8",Concrete,0,-1,0,20,-1,3,0.2,3"""
        
        beam_wall_data = """Beam ID,wall_001
beam_001,1
beam_002,1"""
        
        beam_beam_data = """Beam ID,beam_001,beam_002
beam_001,0,1
beam_002,1,0"""
        
        # Save to temporary files
        temp_dir = "temp_test_files"
        os.makedirs(temp_dir, exist_ok=True)
        
        files_data = {
            "feature_matrix": feature_data,
            "beam_wall_matrix": beam_wall_data,
            "beam_beam_matrix": beam_beam_data
        }
        
        temp_files = {}
        for file_type, data in files_data.items():
            file_path = os.path.join(temp_dir, f"{file_type}.csv")
            with open(file_path, 'w') as f:
                f.write(data)
            temp_files[file_type] = file_path
        
        try:
            # Upload files
            files = {}
            for file_type, file_path in temp_files.items():
                files[file_type] = open(file_path, 'rb')
            
            response = self.session.post(
                f"{self.base_url}/upload/building-data?building_id={building_id}",
                files=files
            )
            
            # Close files
            for f in files.values():
                f.close()
            
            if response.status_code == 200:
                data = response.json()
                print("✅ File upload successful")
                print(f"   Building ID: {data['building_id']}")
                return data["files"]
            else:
                print(f"❌ File upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return {}
                
        except Exception as e:
            print(f"❌ File upload error: {str(e)}")
            return {}
        finally:
            # Cleanup temp files
            for file_path in temp_files.values():
                if os.path.exists(file_path):
                    os.remove(file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    
    def test_stage2(self, file_paths: Dict[str, str]) -> Dict[str, Any]:
        """Test Stage 2 prediction"""
        
        request_data = {
            "building_data": {
                "building_id": "test_building",
                "file_paths": file_paths
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/stage2",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Stage 2 prediction successful")
                print(f"   Job ID: {data['job_id']}")
                print(f"   Total beams: {data['summary']['total_beams']}")
                print(f"   Processing time: {data['processing_time']:.2f}s")
                return data
            else:
                print(f"❌ Stage 2 prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return {}
                
        except Exception as e:
            print(f"❌ Stage 2 prediction error: {str(e)}")
            return {}
    
    def test_stage3(self, file_paths: Dict[str, str], stage2_constraints: Dict[str, int]) -> Dict[str, Any]:
        """Test Stage 3 prediction"""
        
        request_data = {
            "building_data": {
                "building_id": "test_building",
                "file_paths": file_paths
            },
            "stage2_constraints": stage2_constraints
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/stage3",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Stage 3 prediction successful")
                print(f"   Job ID: {data['job_id']}")
                print(f"   Total coordinates: {data['constraint_summary']['total_coordinates']}")
                print(f"   Processing time: {data['processing_time']:.2f}s")
                return data
            else:
                print(f"❌ Stage 3 prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return {}
                
        except Exception as e:
            print(f"❌ Stage 3 prediction error: {str(e)}")
            return {}
    
    def test_full_pipeline(self, file_paths: Dict[str, str]) -> Dict[str, Any]:
        """Test full pipeline"""
        
        request_data = {
            "building_data": {
                "building_id": "test_building",
                "file_paths": file_paths
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/pipeline",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Full pipeline successful")
                print(f"   Job ID: {data['job_id']}")
                print(f"   Stage 2 beams: {data['stage2_results']['summary']['total_beams']}")
                print(f"   Stage 3 coordinates: {data['stage3_results']['constraint_summary']['total_coordinates']}")
                print(f"   Total processing time: {data['total_processing_time']:.2f}s")
                return data
            else:
                print(f"❌ Full pipeline failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return {}
                
        except Exception as e:
            print(f"❌ Full pipeline error: {str(e)}")
            return {}
    
    def test_download(self, job_id: str, file_type: str) -> bool:
        """Test file download"""
        
        try:
            response = self.session.get(f"{self.base_url}/download/{job_id}/{file_type}")
            
            if response.status_code == 200:
                print(f"✅ Download successful: {file_type}")
                print(f"   Content length: {len(response.content)} bytes")
                return True
            else:
                print(f"❌ Download failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Download error: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all API tests"""
        
        print("=" * 60)
        print("STRUCTURAL ANALYSIS PIPELINE API TESTS")
        print("=" * 60)
        
        # Test 1: Health check
        print("\n1. Testing health endpoint...")
        if not self.test_health():
            print("❌ Health check failed - stopping tests")
            return
        
        # Test 2: File upload
        print("\n2. Testing file upload...")
        file_paths = self.upload_test_files()
        if not file_paths:
            print("❌ File upload failed - stopping tests")
            return
        
        # Test 3: Stage 2 prediction
        print("\n3. Testing Stage 2 prediction...")
        stage2_result = self.test_stage2(file_paths)
        if not stage2_result:
            print("❌ Stage 2 failed - skipping Stage 3")
        else:
            # Extract constraints for Stage 3
            stage2_constraints = {}
            for pred in stage2_result.get("predictions", []):
                stage2_constraints[pred["beam_id"]] = pred["predicted_columns"]
            
            # Test 4: Stage 3 prediction
            print("\n4. Testing Stage 3 prediction...")
            stage3_result = self.test_stage3(file_paths, stage2_constraints)
        
        # Test 5: Full pipeline
        print("\n5. Testing full pipeline...")
        pipeline_result = self.test_full_pipeline(file_paths)
        
        # Test 6: Download results
        if pipeline_result and "job_id" in pipeline_result:
            print("\n6. Testing file download...")
            job_id = pipeline_result["job_id"]
            self.test_download(job_id, "pipeline_summary")
        
        print("\n" + "=" * 60)
        print("API TESTS COMPLETED")
        print("=" * 60)

def main():
    """Main test function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Structural Analysis Pipeline API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", choices=["health", "upload", "stage2", "stage3", "pipeline", "all"], 
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.test == "all":
        tester.run_all_tests()
    elif args.test == "health":
        tester.test_health()
    else:
        print(f"Running {args.test} test...")
        # Add individual test implementations as needed

if __name__ == "__main__":
    main()