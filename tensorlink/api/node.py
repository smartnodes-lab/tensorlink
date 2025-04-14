from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException, Depends, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import threading
import uvicorn
import torch


# Use Pydantic models for request validation
class JobRequest(BaseModel):
    model_name: str
    time: int
    payment: int


class GenerationRequest(BaseModel):
    message: str
    max_length: int = 256
    temperature: float = 0.4
    do_sample: bool = True
    history: Optional[List[dict]] = None


# Dependency to verify model is loaded
def get_model():
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=400, detail="No model loaded. Please load a model first."
        )
    return model, tokenizer


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


def create_endpoint(smart_node):
    # Create and start the endpoint
    app = FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5053/api"
        ],  # Update with specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/generate")
    async def generate(
        request: GenerationRequest, model_data: tuple = Depends(get_model)
    ):
        """Generate text from the loaded model"""
        loaded_model, loaded_tokenizer = model_data

        try:
            # logger.info(f"Generating response for input: {request.message[:50]}...")

            # Process input
            input_text = request.message + loaded_tokenizer.eos_token
            inputs = loaded_tokenizer.encode(input_text, return_tensors="pt")

            # Move to appropriate device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = inputs.to(device)

            # Generate response
            with torch.no_grad():
                outputs = loaded_model.generate(
                    inputs,
                    max_length=request.max_length,
                    num_return_sequences=1,
                    temperature=request.temperature,
                    do_sample=request.do_sample,
                    pad_token_id=loaded_tokenizer.eos_token_id,
                )

            # Decode response
            response_text = loaded_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            # logger.info(f"Generation completed: {len(response_text)} characters")

            return {"response": response_text}

        except Exception as e:
            # logger.error(f"Error during generation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/load_model")
    async def load_model(request: JobRequest):
        """Load model and tokenizer"""
        global model, tokenizer

        try:
            # logger.info(f"Loading model: {request.model_name}")

            # Clear GPU memory if applicable
            if model is not None and torch.cuda.is_available():
                del model
                torch.cuda.empty_cache()

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(request.model_name)

            # Load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # logger.info(f"Using device: {device}")

            model = AutoModelForCausalLM.from_pretrained(
                request.model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
            ).to(device)

            # logger.info(f"Model loaded successfully")

            return {
                "success": True,
                "message": f"Model '{request.model_name}' loaded successfully",
            }

        except Exception as e:
            # logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error loading model: {str(e)}"
            )

    @app.post("/api/request-job")
    def request_job(request: Request, job_request: JobRequest):
        client_ip = request.client.host
        smart_node.handle_api_job_req(job_request, client_ip)

    @app.post("/api/node-info")
    def get_node_info():
        response = smart_node.get_self_info()
        return response

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

    thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "0.0.0.0/tensorlink-api", "port": smart_node.port},
    )
    thread.daemon = True
    thread.start()
