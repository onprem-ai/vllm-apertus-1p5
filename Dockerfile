# Exact-commit wheel build for vLLM Apertus 1.5 multimodal serving.
FROM docker.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG VLLM_WHEEL_URL=https://wheels.vllm.ai/3333d7cb6391d27bac146f8eaf869e6e318f429f/vllm-0.26.1rc1.dev164%2Bg3333d7cb6-cp38-abi3-manylinux_2_28_x86_64.whl
ARG TRANSFORMERS_APERTUS_COMMIT=a988895b5160ba13e1a92151a57fd74eb94fd6da

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl patch \
    python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# This CUDA 13.0 wheel was built by upstream vLLM CI from the exact source
# commit used by PR #50496. Installing it avoids rebuilding CUDA extensions.
RUN uv pip install --python /opt/venv/bin/python \
    --torch-backend=cu130 \
    "vllm[audio] @ ${VLLM_WHEEL_URL}"

# Apertus 1.5 AutoProcessor support is pending in Transformers PR #47662.
# Pin the exact reviewed PR head so image and audio processing are reproducible.
RUN uv pip install --python /opt/venv/bin/python \
    --torch-backend=cu130 \
    "transformers @ https://github.com/swiss-ai/transformers/archive/${TRANSFORMERS_APERTUS_COMMIT}.tar.gz"

COPY vendored/apertus_emu35.py /tmp/apertus_emu35.py
COPY vendored/apertus_wavetokenizer.py /tmp/apertus_wavetokenizer.py
COPY vendored/apertus_utils.py /tmp/apertus_utils.py
COPY vendored/apertus_mm.py /tmp/apertus_mm.py
COPY patches/0001-registry.diff /tmp/0001-registry.diff
COPY patches/0002-reasoning-init.diff /tmp/0002-reasoning-init.diff
COPY patches/apertus_reasoning_parser.py /tmp/apertus_reasoning_parser.py

RUN VLLM_SITE_ROOT="$(python3 -c 'from pathlib import Path; import vllm; print(Path(vllm.__file__).parent.parent)')" && \
    VLLM_PACKAGE_DIR="${VLLM_SITE_ROOT}/vllm" && \
    install -m 0644 /tmp/apertus_emu35.py "${VLLM_PACKAGE_DIR}/model_executor/models/apertus_emu35.py" && \
    install -m 0644 /tmp/apertus_wavetokenizer.py "${VLLM_PACKAGE_DIR}/model_executor/models/apertus_wavetokenizer.py" && \
    install -m 0644 /tmp/apertus_utils.py "${VLLM_PACKAGE_DIR}/model_executor/models/apertus_utils.py" && \
    install -m 0644 /tmp/apertus_mm.py "${VLLM_PACKAGE_DIR}/model_executor/models/apertus_mm.py" && \
    install -m 0644 /tmp/apertus_reasoning_parser.py "${VLLM_PACKAGE_DIR}/reasoning/apertus_reasoning_parser.py" && \
    patch -d "${VLLM_SITE_ROOT}" -p1 < /tmp/0001-registry.diff && \
    patch -d "${VLLM_SITE_ROOT}" -p1 < /tmp/0002-reasoning-init.diff

RUN python3 - <<'EOF'
from transformers import Apertus1p5Processor
from vllm.model_executor.models.registry import ModelRegistry

assert Apertus1p5Processor.__name__ == "Apertus1p5Processor"
architectures = ModelRegistry.get_supported_archs()
assert "ApertusForCausalLM" in architectures, "text architecture missing"
assert "Apertus1p5ForConditionalGeneration" in architectures, "multimodal architecture missing"

from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

parser_class = ReasoningParserManager.get_reasoning_parser("apertus")
assert parser_class is not None, "apertus reasoning parser missing"
tool_parser_names = set(ToolParserManager.tool_parsers) | set(
    getattr(ToolParserManager, "lazy_parsers", {})
)
assert "apertus" in tool_parser_names, "apertus tool parser missing"
print("Apertus 1.5 architecture and parsers verified")
EOF

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
