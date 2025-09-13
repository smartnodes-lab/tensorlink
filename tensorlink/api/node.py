from fastapi import FastAPI, HTTPException, APIRouter, Query, Request
from pydantic import BaseModel
from typing import Optional, List
from collections import defaultdict
import threading
import uvicorn
import asyncio
import random
import time


class JobRequest(BaseModel):
    hf_name: str
    time: int
    payment: int


class GenerationRequest(BaseModel):
    hf_name: str
    message: str
    prompt: str = None
    max_length: int = 2048
    temperature: float = 0.4
    do_sample: bool = True
    num_beams: int = 4
    history: Optional[List[dict]] = None
    output: str = None
    processing: bool = False
    id: int = None


class TensorlinkAPI:
    def __init__(self, smart_node, host="0.0.0.0", port=64747):
        self.smart_node = smart_node
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.router = APIRouter()

        self.model_name_to_request = {}
        self.model_request_timestamps = defaultdict(list)

        self._define_routes()
        self._start_server()

    def _define_routes(self):
        @self.router.post("/generate")
        async def generate(request: GenerationRequest):
            print("Incoming request:", request)
            try:
                # Log model request
                current_time = time.time()
                self.model_request_timestamps[request.hf_name].append(current_time)

                cutoff = current_time - 300
                self.model_request_timestamps[request.hf_name] = [
                    ts
                    for ts in self.model_request_timestamps[request.hf_name]
                    if ts > cutoff
                ]

                # Update request counter
                if request.hf_name not in self.model_name_to_request:
                    self.model_name_to_request[request.hf_name] = 0
                self.model_name_to_request[request.hf_name] += 1

                request.output = None
                request.id = hash(random.random())

                # Append model request to the queue
                self.smart_node.endpoint_requests["incoming"].append(request)

                # Wait for the result
                request = await self._wait_for_result(request)

                return_val = request.output
                return {"response": return_val}

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/request-job")
        def request_job(request: Request, job_request: JobRequest):
            # client_ip = request.client.host
            self.smart_node.create_base_job()

        @self.router.get("/api-demand-stats")
        async def get_api_demand_stats():
            """Return current API demand statistics"""
            current_time = time.time()
            demand_stats = {}

            for model_name, timestamps in self.model_request_timestamps.items():
                # Count requests in the last 5 minutes
                recent_requests = sum(
                    1 for ts in timestamps if current_time - ts <= 300
                )
                demand_stats[model_name] = recent_requests

            return demand_stats

        @self.app.get("/stats")
        async def get_network_stats():
            return self.smart_node.get_tensorlink_status()

        @self.app.get("/network-history")
        async def get_network_history(
            days: int = Query(30, ge=1, le=90),
            include_weekly: bool = False,
            include_summary: bool = True,
        ):
            return self.smart_node.get_network_status(
                days=days,
                include_weekly=include_weekly,
                include_summary=include_summary,
            )

        self.app.include_router(self.router)

    async def _wait_for_result(self, request: GenerationRequest) -> GenerationRequest:
        start_time = time.time()
        while True:
            if self.smart_node.endpoint_requests["outgoing"]:
                for response in self.smart_node.endpoint_requests["outgoing"]:
                    if request.id == response.id:
                        return response

            await asyncio.sleep(0.25)
            if time.time() - start_time > 30:
                raise HTTPException(status_code=504, detail="Request timed out.")

    def _start_server(self):
        thread = threading.Thread(
            target=uvicorn.run,
            args=(self.app,),
            kwargs={"host": self.host, "port": self.port},
        )
        thread.daemon = True
        thread.start()

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
