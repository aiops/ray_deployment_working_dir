import inspect
import os
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
import vllm.platforms.cuda

logger = logging.getLogger("ray.serve")

app = FastAPI()

# vLLM has some issues in certain versions, which is why we introduce some additional logic
# https://github.com/vllm-project/vllm/issues/7890
# https://github.com/vllm-project/vllm/issues/8402
# Goal: If possible not invasive, non-blocking
# Save a reference to the original function
original_function = vllm.platforms.cuda.device_id_to_physical_device_id
def device_id_to_physical_device_id_wrapper(*args, **kwargs):
    logger.info(f"Hook: Executing code before calling "
                    f"'device_id_to_physical_device_id' (with args={args}, kwargs={kwargs}).")
    if not len(os.environ["CUDA_VISIBLE_DEVICES"]):
        try:
            import nvsmi
            gpu_count: int = len(list(nvsmi.get_gpus()))
            new_env_value: str = ",".join([str(n) for n in range(gpu_count)])
            os.environ["CUDA_VISIBLE_DEVICES"] = new_env_value
            logger.info(f"New value for environment variable 'CUDA_VISIBLE_DEVICES': {new_env_value}")
        except BaseException as e:
            logger.error(f"Could not derive gpu_count using 'nvsmi' library. Error: {e}")
    func_response = original_function(*args, **kwargs)
    logger.info(f"function 'device_id_to_physical_device_id' response: {func_response}")
    return func_response
# Replace the original function with the wrapped version
vllm.platforms.cuda.device_id_to_physical_device_id = device_id_to_physical_device_id_wrapper

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
        logger.info(f"Downloaded {file_name} to {file_path}")
    else:
        logger.info(f"{file_name} already exists at {file_path}")
    # Return the new file path
    return str(file_path)

def get_base_model_paths(engine_args: AsyncEngineArgs, target_clazz) -> list:
    def _has_parameter(t_clazz, param_name):
        # Get the signature of the class's __init__ method
        init_signature = inspect.signature(t_clazz.__init__)
        # Check each parameter in the signature
        for name, param in init_signature.parameters.items():
            if name == param_name:
                return True
        # If the parameter is not found, it's not required
        return False

    if engine_args.served_model_name is not None:
        served_model_names = engine_args.served_model_name
    else:
        served_model_names = [engine_args.model]

    if _has_parameter(target_clazz, "base_model_paths"):
        from vllm.entrypoints.openai.serving_engine import BaseModelPath
        base_model_paths = [
            BaseModelPath(name=name, model_path=engine_args.model)
            for name in served_model_names
        ]
        return base_model_paths
    elif _has_parameter(target_clazz, "served_model_names"):
        return served_model_names
    else:
        logger.info("Should not happen!")

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
        self.openai_serving_chat = None
        self.openai_serving_completion = None
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.chat_template = chat_template
        engine_args.model = download_gguf_file(engine_args.model)
        logger.info(f"Starting with engine args: {engine_args}")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.engine_args = engine_args

    @app.post("/completions")
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
            base_model_paths_or_served_model_names = get_base_model_paths(self.engine_args, OpenAIServingCompletion)
            self.openai_serving_completion = OpenAIServingCompletion(
                self.engine,
                model_config,
                base_model_paths_or_served_model_names,
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

    @app.post("/chat/completions")
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
            base_model_paths_or_served_model_names = get_base_model_paths(self.engine_args, OpenAIServingChat)
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                base_model_paths_or_served_model_names,
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

    if tp > 1:
        return VLLMDeployment.options(
            placement_group_bundles=pg_resources,
            placement_group_strategy=os.environ.get('BUILD_APP_ARG_PLACEMENT_GROUP_STRATEGY', "PACK")
        ).bind(
            engine_args,
            parsed_args.response_role,
            parsed_args.lora_modules,
            parsed_args.chat_template,
        )
    else:
        return VLLMDeployment.bind(
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

model = build_app(dynamic_ray_engine_args)