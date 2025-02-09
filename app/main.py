import ray
from ray import serve
from app.core.service import WhisperService

if __name__ == "__main__":
    # Initialize Ray in detached mode
    ray.init(address="auto", namespace="serve", runtime_env={"working_dir": "."})
    
    # Start Ray Serve
    serve.start(detached=True, http_options={
        "host": "0.0.0.0",
        "port": 8000,
        "request_timeout_s": 300  # 5 minutes timeout for long transcriptions
    })
    
    # Deploy the service
    serve.run(WhisperService.bind(), route_prefix="/api/v1") 