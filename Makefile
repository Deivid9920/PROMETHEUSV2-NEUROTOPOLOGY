# Prometheus-NS build orchestration.

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
TORCH_CPU_INDEX := https://download.pytorch.org/whl/cpu
PROFILE ?= auto          # auto probes the hardware and adapts (make autofit previews it)
TOKENS ?= 5.0e6
ROUNDS ?= 1

AUTO_FLAG :=
ifeq ($(AUTO_CONTINUE),1)
AUTO_FLAG := --auto-continue
endif

# Default chat path is dynamic int8; QUANT=0 selects the fp32 reference.
QUANT_FLAG := $(if $(filter 0,$(QUANT)),--fp32,--int8)

.PHONY: setup setup-gpu autofit data tokenize train eval loop chat test lint docker-build clean

# CLI contracts fixed by this Makefile (implementations must honor them):
#   prometheus_ns.data.crawler   --config PATH
#   prometheus_ns.data.cleaner   --config PATH
#   prometheus_ns.data.reporter  --config PATH
#   prometheus_ns.model.tokenizer_train --config PATH
#   prometheus_ns.train.trainer  --config PATH --profile NAME --max-tokens N
#     (NAME accepts auto: the hardware is probed and the profile adapts)
#   prometheus_ns.autoloop.loop  --config PATH --rounds N [--auto-continue]
#   prometheus_ns.eval.perplexity --config PATH
#   prometheus_ns.eval.prompt_suite --config PATH
#   scripts/chat.py              --config PATH (--int8 | --fp32)

setup:
	test -f requirements.txt || { echo "requirements.txt missing: create it first"; exit 1; }
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install "torch>=2.3,<2.5" --index-url $(TORCH_CPU_INDEX)
	$(PIP) install -r requirements.txt
	$(PY) -m spacy download en_core_web_sm

setup-gpu:
	test -f requirements-gpu.txt || { echo "requirements-gpu.txt missing: see docs/gpu_migration.md"; exit 1; }
	$(PIP) install -r requirements-gpu.txt

data:
	$(PY) -m prometheus_ns.data.crawler  --config config.yaml
	$(PY) -m prometheus_ns.data.cleaner  --config config.yaml
	$(PY) -m prometheus_ns.data.reporter --config config.yaml

tokenize:
	$(PY) -m prometheus_ns.model.tokenizer_train --config config.yaml

autofit:
	$(PY) -m prometheus_ns.autofit --config config.yaml

train:
	$(PY) -m prometheus_ns.train.trainer --config config.yaml --profile $(PROFILE) --max-tokens $(TOKENS)

eval:
	$(PY) -m prometheus_ns.eval.perplexity   --config config.yaml
	$(PY) -m prometheus_ns.eval.prompt_suite --config config.yaml

loop:
	$(PY) -m prometheus_ns.autoloop.loop --config config.yaml --rounds $(ROUNDS) $(AUTO_FLAG)

chat:
	$(PY) scripts/chat.py --config config.yaml $(QUANT_FLAG)

test:
	$(PY) -m pytest tests -v

lint:
	$(PY) -m compileall -q prometheus_ns scripts
	@command -v ruff >/dev/null 2>&1 && ruff check prometheus_ns scripts || echo "ruff not installed: syntax check only"

docker-build:
	@test -f Dockerfile || { echo "Dockerfile not present yet"; exit 1; }
	docker build -t prometheus-ns .

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +

MODE ?= dup_flood
RID ?= 1

topo-extract:
	$(PY) scripts/run_round_v2.py --config config.yaml --topo-only

topo-graph:
	$(PY) -c "from prometheus_ns.topo.graph_topology import snapshot_current; snapshot_current('config.yaml')"

topo: topo-extract topo-graph

stress:
	$(PY) scripts/run_round_v2.py --config config.yaml --stress $(MODE) --round-id $(RID)

divergence-report:
	$(PY) scripts/divergence_report.py --config config.yaml
