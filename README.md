# Structural Analysis Pipeline API

Production FastAPI service for Stage 2 (Column Count Prediction) and Stage 3 (Coordinate Prediction) machine learning pipelines.

## Features

- **Stage 2 API**: Graph Neural Network for column count prediction
- **Stage 3 API**: CNN for column coordinate prediction with constraints
- **Full Pipeline**: Combined Stage 2 + Stage 3 with constraint enforcement
- **File Upload/Download**: CSV file handling for building data
- **Job Management**: Asynchronous job tracking and results storage
- **Production Ready**: Docker containerization, health checks, logging

## Quick Start

### 1. Setup Models

Copy your trained models to the `models/` directory:

```bash
mkdir -p ProductionAPI/models
cp stage2_model.pth ProductionAPI/models/
cp column_predictor_no_leakage.pth ProductionAPI/models/
```

### 2. Run with Docker

```bash
cd ProductionAPI
docker-compose up --build
```

The API will be available at `http://localhost:8000`

### 3. Run Locally

```bash
cd ProductionAPI
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API

```bash
# Test original functionality
python test_api.py

# Test new CSV processing functionality  
python test_csv_processing.py
```

## API Endpoints

### Health Check
```http
GET /health
```

### CSV Processing (New!)
```http
POST /process-csv?building_id=2024332
Content-Type: multipart/form-data

csv_file: [Single CSV file with all building elements]
```

This endpoint takes a single CSV file containing all building elements (beams, columns, walls) and automatically:
- Detects geometric connections between elements
- Generates BeamWallMatrix, BeamBeamMatrix, BeamColumnMatrix
- Creates FeatureMatrix with all elements
- Returns file paths for use in pipeline

### CSV Processing + Stage 2 Inference (New!)
```http
POST /process-csv-with-stage2?building_id=2024332
Content-Type: multipart/form-data

csv_file: [Single CSV file with all building elements]
```

This endpoint processes CSV and immediately runs Stage 2 model inference:
- Generates connection matrices from CSV
- Runs Stage 2 Graph Neural Network
- Returns JSON response with column count predictions for each beam
- Includes confidence scores and processing statistics

### CSV Processing + Full Pipeline (New!)
```http
POST /process-csv-and-predict?building_id=2024332
Content-Type: multipart/form-data

csv_file: [Single CSV file with all building elements]
```

Convenience endpoint that processes CSV and immediately runs the full pipeline.

### Stage 2: Column Count Prediction
```http
POST /predict/stage2
Content-Type: application/json

{
  "building_data": {
    "building_id": "2024332",
    "file_paths": {
      "feature_matrix": "/path/to/FeatureMatrix.csv",
      "beam_wall_matrix": "/path/to/BeamWallMatrix.csv",
      "beam_beam_matrix": "/path/to/BeamBeamMatrix.csv"
    }
  }
}
```

### Stage 3: Coordinate Prediction
```http
POST /predict/stage3
Content-Type: application/json

{
  "building_data": {
    "building_id": "2024332",
    "file_paths": {...}
  },
  "stage2_constraints": {
    "beam_001": 1,
    "beam_002": 2,
    "beam_003": 0
  }
}
```

### Full Pipeline
```http
POST /predict/pipeline
Content-Type: application/json

{
  "building_data": {
    "building_id": "2024332",
    "file_paths": {...}
  }
}
```

### File Upload
```http
POST /upload/building-data?building_id=2024332
Content-Type: multipart/form-data

feature_matrix: [CSV file]
beam_wall_matrix: [CSV file]
beam_beam_matrix: [CSV file]
beam_column_matrix: [CSV file] (optional)
```

### Download Results
```http
GET /download/{job_id}/{file_type}
```

File types:
- `stage2_predictions`
- `stage3_coordinates`
- `pipeline_summary`

## API Documentation

Once running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Example Usage

### Python Client

```python
import requests

# Method 1: Single CSV Processing
with open('building_data.csv', 'rb') as f:
    files = {'csv_file': f}
    csv_response = requests.post(
        'http://localhost:8000/process-csv?building_id=2024332',
        files=files
    )

csv_result = csv_response.json()
print(f"Generated matrices: {list(csv_result['file_paths'].keys())}")

# Method 2: CSV Processing + Stage 2 Inference Only
with open('building_data.csv', 'rb') as f:
    files = {'csv_file': f}
    stage2_response = requests.post(
        'http://localhost:8000/process-csv-with-stage2?building_id=2024332',
        files=files
    )

stage2_result = stage2_response.json()
print(f"Stage 2 Predictions:")
for pred in stage2_result['stage2_predictions']:
    print(f"  {pred['beam_id']}: {pred['predicted_columns']} columns (confidence: {pred['confidence']:.3f})")

# Method 3: CSV Processing + Full Pipeline (One-step)
with open('building_data.csv', 'rb') as f:
    files = {'csv_file': f}
    pipeline_response = requests.post(
        'http://localhost:8000/process-csv-and-predict?building_id=2024332',
        files=files
    )

result = pipeline_response.json()
print(f"Total coordinates predicted: {result['stage3_results']['constraint_summary']['total_coordinates']}")

# Method 4: Traditional Multi-file Upload (Original)
files = {
    'feature_matrix': open('2024332_FeatureMatrix.csv', 'rb'),
    'beam_wall_matrix': open('2024332_BeamWallMatrix.csv', 'rb'),
    'beam_beam_matrix': open('2024332_BeamBeamMatrix.csv', 'rb')
}

upload_response = requests.post(
    'http://localhost:8000/upload/building-data?building_id=2024332',
    files=files
)

# Run full pipeline
pipeline_request = {
    "building_data": {
        "building_id": "2024332",
        "file_paths": upload_response.json()["files"]
    }
}

pipeline_response = requests.post(
    'http://localhost:8000/predict/pipeline',
    json=pipeline_request
)

result = pipeline_response.json()
print(f"Job ID: {result['job_id']}")
print(f"Total coordinates predicted: {result['stage3_results']['constraint_summary']['total_coordinates']}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# CSV Processing
curl -X POST "http://localhost:8000/process-csv?building_id=2024332" \
  -F "csv_file=@building_data.csv"

# CSV Processing + Stage 2 Inference
curl -X POST "http://localhost:8000/process-csv-with-stage2?building_id=2024332" \
  -F "csv_file=@building_data.csv"

# CSV Processing + Full Pipeline
curl -X POST "http://localhost:8000/process-csv-and-predict?building_id=2024332" \
  -F "csv_file=@building_data.csv"

# Traditional file upload
curl -X POST "http://localhost:8000/upload/building-data?building_id=2024332" \
  -F "feature_matrix=@2024332_FeatureMatrix.csv" \
  -F "beam_wall_matrix=@2024332_BeamWallMatrix.csv" \
  -F "beam_beam_matrix=@2024332_BeamBeamMatrix.csv"
```

## CSV Input Format

The new CSV processing endpoints expect a single CSV file with the following columns:

### Required Columns
- `Element ID`: Unique identifier for each element
- `Element Type`: "Structural Framing", "Structural Column", or "Wall"
- `Family`: Element family name
- `Structural Material`: Material type (must contain "Metal" or "Steel" for beams)
- `Start X`, `Start Y`, `Start Z`: Starting coordinates
- `End X`, `End Y`, `End Z`: Ending coordinates
- `Width`, `Height`: Element dimensions

### Additional Columns (for walls)
- `Entity Start Level`, `Entity End Level`: Vertical levels for walls

### Sample CSV Structure
```csv
Element ID,Element Type,Family,Structural Material,Start X,Start Y,Start Z,End X,End Y,End Z,Width,Height,Entity Start Level,Entity End Level
beam_001,Structural Framing,W12x26,Steel 43-275,0,0,10,20,0,10,0.5,1.0,,
col_001,Structural Column,HSS8x8x1/2,Steel 43-275,20,0,0,20,0,15,0.8,0.8,,
wall_001,Wall,Generic - 8",Concrete,0,-1,0,40,-1,0,0.67,12,0,12
```

## Connection Detection Logic

The API automatically detects connections using geometric analysis:

### Beam-Beam Connections
- Checks if beam endpoints are within intersection tolerance of other beams
- Considers vertical overlap between beams
- Uses beam width for proximity calculations

### Beam-Column Connections  
- Detects when beam endpoints are near column centers
- Ensures beam hits upper half of column (not column base)
- Prioritizes column connections over wall connections

### Beam-Wall Connections
- Only detected if no column connection exists at beam endpoint
- Uses point-to-line-segment distance calculation
- Considers wall thickness and vertical overlap

### Tolerance Parameters
- `INTERSECTION_TOL`: 1.0 feet (horizontal tolerance)
- `VERTICAL_BEAM_TOL`: 0.4 feet (beam vertical overlap)
- `VERTICAL_COLUMN_TOL`: 3.0 feet (column vertical overlap)
- `VERTICAL_WALL_TOL`: 0.3 feet (wall vertical overlap)
- `COLUMN_RADIUS`: 0.4 feet (column connection radius)

## Configuration

### Environment Variables

- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MODEL_PATH_STAGE2`: Path to Stage 2 model file
- `MODEL_PATH_STAGE3`: Path to Stage 3 model file
- `TEMP_DATA_DIR`: Directory for temporary files
- `MAX_FILE_AGE_HOURS`: Auto-cleanup age for temporary files

### Model Paths

Update model paths in the respective model classes:

```python
# app/models/stage2_model.py
class Stage2Predictor:
    def __init__(self, model_path: str = "models/stage2_model.pth"):

# app/models/stage3_model.py  
class Stage3Predictor:
    def __init__(self, model_path: str = "models/column_predictor_no_leakage.pth"):
```

## Production Deployment

### Docker Deployment

1. **Build and run**:
```bash
docker-compose up -d
```

2. **Scale services**:
```bash
docker-compose up -d --scale structural-analysis-api=3
```

3. **Monitor logs**:
```bash
docker-compose logs -f structural-analysis-api
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: structural-analysis-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: structural-analysis-api
  template:
    metadata:
      labels:
        app: structural-analysis-api
    spec:
      containers:
      - name: api
        image: structural-analysis-api:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi" 
            cpu: "2"
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

## Performance Considerations

### Hardware Requirements

- **CPU**: Minimum 4 cores, recommended 8+ cores
- **Memory**: Minimum 8GB RAM, recommended 16GB+
- **GPU**: Optional but recommended for faster inference
- **Storage**: SSD recommended for model loading and file I/O

### Optimization Tips

1. **GPU Acceleration**: Ensure CUDA is available for PyTorch
2. **Batch Processing**: Process multiple buildings in batches
3. **Caching**: Implement Redis caching for frequent requests
4. **Load Balancing**: Use multiple API instances behind a load balancer
5. **File Cleanup**: Regular cleanup of temporary files

## Monitoring

### Health Checks

The API includes built-in health checks:
- `/health` - Basic health status
- Docker health check every 30 seconds
- Model loading status verification

### Logging

Structured logging with different levels:
- **INFO**: Normal operations, job completion
- **WARNING**: Non-critical issues, missing data
- **ERROR**: Failures, exceptions
- **DEBUG**: Detailed debugging information

### Metrics

Consider adding:
- Prometheus metrics for monitoring
- Request/response times
- Model inference times
- Error rates
- Resource usage

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check model file paths
   - Verify model file permissions
   - Ensure sufficient memory

2. **CUDA Out of Memory**
   - Reduce batch sizes
   - Use CPU inference
   - Add more GPU memory

3. **File Upload Errors**
   - Check file size limits
   - Verify CSV format
   - Ensure proper column names

4. **Slow Performance**
   - Enable GPU acceleration
   - Optimize model loading
   - Use SSD storage

### Debug Mode

Run with debug logging:
```bash
LOG_LEVEL=DEBUG uvicorn app.main:app --reload
```

## Development

### Project Structure

```
ProductionAPI/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/              # ML model wrappers
│   │   ├── stage2_model.py  # Stage 2 GNN
│   │   ├── stage3_model.py  # Stage 3 CNN
│   │   └── pipeline_manager.py
│   └── utils/               # Utilities
│       ├── response_models.py
│       └── file_handler.py
├── models/                  # Trained model files
├── temp_data/              # Temporary storage
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### Adding New Features

1. **New Endpoints**: Add to `app/main.py`
2. **Model Updates**: Modify model classes in `app/models/`
3. **Response Models**: Update `app/utils/response_models.py`
4. **File Handling**: Extend `app/utils/file_handler.py`

## License

This API is part of the Structural Analysis Pipeline project.