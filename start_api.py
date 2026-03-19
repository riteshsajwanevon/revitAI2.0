#!/usr/bin/env python3
"""
Production API Startup Script
"""

import uvicorn
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Start the FastAPI server"""
    
    # Add current directory to Python path
    import sys
    sys.path.insert(0, os.path.abspath('.'))
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting Structural Analysis Pipeline API")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Log Level: {log_level}")
    
    # Check if model files exist
    model_files = [
        "models/stage2_model.pth",
        "models/column_predictor_no_leakage.pth"
    ]
    
    missing_models = []
    for model_file in model_files:
        if not os.path.exists(model_file):
            missing_models.append(model_file)
    
    if missing_models:
        logger.warning(f"Missing model files: {missing_models}")
        logger.warning("API will start but model loading may fail")
    
    # Start server
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
            reload=False,
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()