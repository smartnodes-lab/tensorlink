from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Literal


class NodeRequest(BaseModel):
    address: str


class JobRequest(BaseModel):
    hf_name: str
    model_type: Optional[str] = None
    time: int = 1800
    payment: int = 0


class GenerationRequest(BaseModel):
    hf_name: str
    message: str
    prompt: str = None
    model_type: Optional[str] = "auto"
    max_length: int = 2048
    max_new_tokens: int = 2048
    temperature: float = 0.4
    do_sample: bool = True
    num_beams: int = 4
    history: Optional[List[dict]] = None
    output: str = None
    processing: bool = False
    id: int = None
    response_format: Literal["simple", "openai", "full"] = "full"


class ModelStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    status: str  # "loaded", "loading", "not_loaded", "error"
    message: str
