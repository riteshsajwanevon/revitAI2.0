#!/usr/bin/env python3
"""
Example: CSV Processing with Stage 2 Inference
Demonstrates how to use the new CSV processing + Stage 2 endpoint
"""

import requests
import pandas as pd
import json
import tempfile
import os

def create_example_building_csv():
    """Create a sample building CSV file"""
    
    # Sample building data - a simple 2-story structure
    building_data = [
        # Ground floor beams
        {
            "Element ID": "B001", "Element Type": "Structural Framing", "Family": "W14x30",
            "Structural Material": "Steel 50-355", "Start X": 0, "Start Y": 0, "Start Z": 12,
            "End X": 30, "End Y": 0, "End Z": 12, "Width": 0.6, "Height": 1.4
        },
        {
            "Element ID": "B002", "Element Type": "Structural Framing", "Family": "W14x30", 
            "Structural Material": "Steel 50-355", "Start X": 30, "Start Y": 0, "Start Z": 12,
            "End X": 60, "End Y": 0, "End Z": 12, "Width": 0.6, "Height": 1.4
        },
        {
            "Element ID": "B003", "Element Type": "Structural Framing", "Family": "W14x30",
            "Structural Material": "Steel 50-355", "Start X": 0, "Start Y": 20, "Start Z": 12,
            "End X": 30, "End Y": 20, "End Z": 12, "Width": 0.6, "Height": 1.4
        },
        {
            "Element ID": "B004", "Element Type": "Structural Framing", "Family": "W14x30",
            "Structural Material": "Steel 50-355", "Start X": 30, "Start Y": 20, "Start Z": 12,
            "End X": 60, "End Y": 20, "End Z": 12, "Width": 0.6, "Height": 1.4
        },
        
        # Second floor beams
        {
            "Element ID": "B005", "Element Type": "Structural Framing", "Family": "W12x26",
            "Structural Material": "Steel 50-355", "Start X": 0, "Start Y": 0, "Start Z": 24,
            "End X": 30, "End Y": 0, "End Z": 24, "Width": 0.5, "Height": 1.2
        },
        {
            "Element ID": "B006", "Element Type": "Structural Framing", "Family": "W12x26",
            "Structural Material": "Steel 50-355", "Start X": 30, "Start Y": 0, "Start Z": 24,
            "End X": 60, "End Y": 0, "End Z": 24, "Width": 0.5, "Height": 1.2
        },
        
        # Columns
        {
            "Element ID": "C001", "Element Type": "Structural Column", "Family": "HSS10x10x1/2",
            "Structural Material": "Steel 50-355", "Start X": 30, "Start Y": 0, "Start Z": 0,
            "End X": 30, "End Y": 0, "End Z": 30, "Width": 0.83, "Height": 0.83
        },
        {
            "Element ID": "C002", "Element Type": "Structural Column", "Family": "HSS10x10x1/2", 
            "Structural Material": "Steel 50-355", "Start X": 30, "Start Y": 20, "Start Z": 0,
            "End X": 30, "End Y": 20, "End Z": 30, "Width": 0.83, "Height": 0.83
        },
        {
            "Element ID": "C003", "Element Type": "Structural Column", "Family": "HSS8x8x1/2",
            "Structural Material": "Steel 50-355", "Start X": 60, "Start Y": 10, "Start Z": 0,
            "End X": 60, "End Y": 10, "End Z": 15, "Width": 0.67, "Height": 0.67
        },
        
        # Walls
        {
            "Element ID": "W001", "Element Type": "Wall", "Family": "Generic - 8\"",
            "Structural Material": "Concrete", "Start X": 0, "Start Y": -1, "Start Z": 0,
            "End X": 60, "End Y": -1, "End Z": 0, "Width": 0.67, "Height": 12,
            "Entity Start Level": 0, "Entity End Level": 12
        },
        {
            "Element ID": "W002", "Element Type": "Wall", "Family": "Generic - 8\"",
            "Structural Material": "Concrete", "Start X": 0, "Start Y": 21, "Start Z": 0,
            "End X": 60, "End Y": 21, "End Z": 0, "Width": 0.67, "Height": 12,
            "Entity Start Level": 0, "Entity End Level": 12
        },
        {
            "Element ID": "W003", "Element Type": "Wall", "Family": "Generic - 6\"",
            "Structural Material": "Concrete", "Start X": -1, "Start Y": 0, "Start Z": 0,
            "End X": -1, "End Y": 20, "End Z": 0, "Width": 0.5, "Height": 12,
            "Entity Start Level": 0, "Entity End Level": 12
        }
    ]
    
    # Create DataFrame and save to temporary file
    df = pd.DataFrame(building_data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name

def test_csv_stage2_endpoint(api_url="http://localhost:8000"):
    """Test the CSV processing + Stage 2 inference endpoint"""
    
    print("🏗️  Creating sample building CSV...")
    csv_file_path = create_example_building_csv()
    
    try:
        print(f"📤 Uploading CSV and running Stage 2 inference...")
        
        # Upload CSV and run Stage 2 inference
        with open(csv_file_path, 'rb') as f:
            files = {'csv_file': f}
            response = requests.post(
                f"{api_url}/process-csv-with-stage2?building_id=example_building_2024",
                files=files
            )
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Success! Stage 2 inference completed.")
            print(f"🆔 Job ID: {result['job_id']}")
            print(f"🏢 Building ID: {result['building_id']}")
            print(f"⏱️  Processing Time: {result['processing_time']:.2f} seconds")
            
            print("\n📊 Connection Summary:")
            for key, value in result['connection_summary'].items():
                print(f"   {key.replace('_', ' ').title()}: {value}")
            
            print("\n📈 Stage 2 Model Results:")
            print(f"   Total Beams Analyzed: {result['stage2_summary']['total_beams']}")
            print(f"   Average Confidence: {result['stage2_summary']['average_confidence']:.3f}")
            print(f"   Predictions by Column Count:")
            for count, num_beams in result['stage2_summary']['predictions_by_count'].items():
                print(f"     {count} columns: {num_beams} beams")
            
            print("\n🔍 Individual Beam Predictions:")
            print("   Beam ID    | Predicted Columns | Confidence | Material")
            print("   " + "-" * 55)
            
            for pred in result['stage2_predictions']:
                material = pred.get('material_prediction', 'N/A')
                print(f"   {pred['beam_id']:<10} | {pred['predicted_columns']:<17} | {pred['confidence']:<10.3f} | {material}")
            
            print(f"\n💾 Generated Files:")
            for file_type, file_path in result['file_paths'].items():
                print(f"   {file_type}: {file_path}")
            
            # Save results to JSON file
            output_file = "stage2_results.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n📁 Results saved to: {output_file}")
            
            return result
            
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the API server is running at", api_url)
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None
    finally:
        # Cleanup temporary file
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)

def main():
    """Main function"""
    print("=" * 70)
    print("CSV PROCESSING + STAGE 2 INFERENCE EXAMPLE")
    print("=" * 70)
    
    # Test the endpoint
    result = test_csv_stage2_endpoint()
    
    if result:
        print("\n🎉 Example completed successfully!")
        print("The API processed the CSV file, detected connections, and ran Stage 2 inference.")
        print("Check the generated 'stage2_results.json' file for detailed results.")
    else:
        print("\n💥 Example failed. Please check the API server and try again.")

if __name__ == "__main__":
    main()