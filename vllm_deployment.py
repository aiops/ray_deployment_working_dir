import os
import urllib
from typing import Dict, Optional, List
import logging
import pathlib
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from huggingface_hub import hf_hub_download

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.utils import FlexibleArgumentParser

logger = logging.getLogger("ray.serve")

app = FastAPI()

def download_gguf_file(model_name_or_path: str) -> str:
    # Only proceed if the URL ends with .gguf
    if not model_name_or_path.endswith(".gguf"):
        logger.info("File does not have .gguf suffix, skipping download.")
        return model_name_or_path  # Return original URL if not a .gguf file
    # Define download path
    download_path = pathlib.Path("/tmp/models")
    download_path.mkdir(parents=True, exist_ok=True)
    # Extract file name and define full download path
    file_name = pathlib.Path(model_name_or_path).name
    file_path = download_path.joinpath(file_name)
    repo_id = str(pathlib.Path(model_name_or_path).parent)
    # Download the file if it doesn't already exist
    if not file_path.exists():
        hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=download_path)
        logger.info(f"Downloaded {file_name} to {file_path}, or retrieved cache version")
    else:
        logger.info(f"{file_name} already exists at {file_path}")
    # Return the new file path
    return str(file_path)


@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
            self,
            engine_args: AsyncEngineArgs,
            response_role: str,
            lora_modules: Optional[List[LoRAModulePath]] = None,
            chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.openai_serving_completion = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/completions")
    async def create_completion(
            self, request: CompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_completion:
            model_config = await self.engine.get_model_config()
            # Determine the name of the served model for the OpenAI client.
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            self.openai_serving_completion = OpenAIServingCompletion(
                self.engine,
                model_config,
                served_model_names,
                lora_modules=self.lora_modules,
                prompt_adapters=None,
                request_logger=None
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_completion.create_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        assert isinstance(generator, CompletionResponse)
        return JSONResponse(content=generator.model_dump())

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
            self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            # Determine the name of the served model for the OpenAI client.
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                served_model_names,
                self.response_role,
                lora_modules=self.lora_modules,
                prompt_adapters=None,
                request_logger=None,
                chat_template=self.chat_template,
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    
    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for key, value in cli_args.items():
        if value is not None:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    tp = engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    cpu_per_actor = int(os.environ["BUILD_APP_ARG_CPU_PER_ACTOR"])
    gpu_per_actor = int(os.environ["BUILD_APP_ARG_GPU_PER_ACTOR"])
    for _ in range(tp):
        pg_resources.append({"CPU": cpu_per_actor, "GPU": gpu_per_actor})  # for the vLLM actors

    return VLLMDeployment.options(
        placement_group_bundles=pg_resources,
        placement_group_strategy=os.environ['BUILD_APP_ARG_PLACEMENT_GROUP_STRATEGY']
    ).bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.chat_template,
    )

# Initialize an empty dictionary
dynamic_ray_engine_args = {}
# Iterate over all environment variables
for key, value in os.environ.items():
    # Check if the environment variable starts with the prefix "DYNAMIC_RAY_CLI_ARG"
    if key.startswith("DYNAMIC_RAY_CLI_ARG") and value is not None and len(value):
        # Remove the prefix, convert to lowercase, and replace underscores with hyphens
        processed_key = key[len("DYNAMIC_RAY_CLI_ARG_"):].lower().replace("_", "-")
        # Add the processed key and its value to the dictionary
        dynamic_ray_engine_args[processed_key] = value

dynamic_ray_engine_args["model"] = download_gguf_file(dynamic_ray_engine_args["model"])
model = build_app(dynamic_ray_engine_args)