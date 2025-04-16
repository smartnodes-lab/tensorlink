from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException, APIRouter, Depends, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import threading
import uvicorn
import torch


# For long term inference model requests
class JobRequest(BaseModel):
    _model_name: str
    time: int
    payment: int


# Handle inference requests for loaded models
class GenerationRequest(BaseModel):
    _model_name: str
    message: str
    max_length: int = 256
    temperature: float = 0.4
    do_sample: bool = True
    num_beams: int = 4
    history: Optional[List[dict]] = None


def create_endpoint(smart_node, port):
    # Create and start the endpoint
    app = FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://127.0.0.1:{port}/tensorlink-api"
        ],  # Update with specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = APIRouter(prefix="/tensorlink-api")

    @router.post("/generate")
    async def generate(request: GenerationRequest):
        """Generate text from a loaded model"""
        try:
            # Process input
            input_text = request.message

        except Exception as e:
            # logger.error(f"Error during generation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    app.include_router(router)
    thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={
            "host": "127.0.0.1" if smart_node.local_test else "0.0.0.0",
            "port": port,
        },
    )
    thread.daemon = True
    thread.start()

    return thread
    # @app.post("/api/load_model")
    # async def load_model(request: JobRequest):
    #     """Load model and tokenizer"""
    #     global model, tokenizer
    #
    #     try:
    #         # logger.info(f"Loading model: {request.model_name}")
    #
    #         # Clear GPU memory if applicable
    #         if model is not None and torch.cuda.is_available():
    #             del model
    #             torch.cuda.empty_cache()
    #
    #         # Load tokenizer
    #         tokenizer = AutoTokenizer.from_pretrained(request.model_name)
    #
    #         # Load model
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         # logger.info(f"Using device: {device}")
    #
    #         model = AutoModelForCausalLM.from_pretrained(
    #             request.model_name,
    #             torch_dtype=(
    #                 torch.float16 if torch.cuda.is_available() else torch.float32
    #             ),
    #         ).to(device)
    #
    #         # logger.info(f"Model loaded successfully")
    #
    #         return {
    #             "success": True,
    #             "message": f"Model '{request.model_name}' loaded successfully",
    #         }
    #
    #     except Exception as e:
    #         # logger.error(f"Error loading model: {str(e)}")
    #         raise HTTPException(
    #             status_code=500, detail=f"Error loading model: {str(e)}"
    #         )
    #
    # @app.get("/api/status")
    # async def get_status():
    #     """Get current model status"""
    #     model_loaded = model is not None and tokenizer is not None
    #     node_active = node is not None
    #
    #     device_info = {}
    #     if torch.cuda.is_available():
    #         device_info = {
    #             "gpu_available": True,
    #             "device_name": torch.cuda.get_device_name(0),
    #             "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB",
    #             "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB"
    #         }
    #     else:
    #         device_info = {"gpu_available": False}
    #
    #     return {
    #         "model_loaded": model_loaded,
    #         "model_name": getattr(model, "name_or_path", None) if model else None,
    #         "tensorlink_active": node_active,
    #         "device_info": device_info
    #     }
    #
    #
    # @app.post("/api/unload_model")
    # async def unload_model():
    #     """Unload the current model to free resources"""
    #     global model, tokenizer
    #
    #     try:
    #         if model is not None:
    #             # logger.info("Unloading model...")
    #             del model
    #             model = None
    #
    #             if torch.cuda.is_available():
    #                 torch.cuda.empty_cache()
    #
    #             tokenizer = None
    #             # logger.info("Model unloaded successfully")
    #
    #         return {"success": True, "message": "Model unloaded successfully"}
    #
    #     except Exception as e:
    #         # logger.error(f"Error unloading model: {str(e)}")
    #         raise HTTPException(status_code=500, detail=str(e))
    #
    # @app.post("/api/request-job")
    # def request_job(request: Request, job_request: JobRequest):
    #     client_ip = request.client.host
    #     smart_node.handle_api_job_req(job_request, client_ip)
    #
    # @app.post("/api/node-info")
    # def get_node_info():
    #     response = smart_node.get_self_info()
    #     return response
    #
    # @app.post("/jobs", methods=["POST"])
    # def upload_job_info():
    #     data = request.get_json()
    #     job_id = data.get("job_id")
    #     job_info = data.get("job_info")
    #     smart_node.jobs.append({job_id: job_info})
    #     return (
    #         jsonify({"message": "Job info uploaded successfully", "job_id": job_id}),
    #         200,
    #     )
    #
    # return app
