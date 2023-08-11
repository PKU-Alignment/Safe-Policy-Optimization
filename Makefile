print-%  : ; @echo $* = $($*)
PROJECT_NAME   = safepo
COPYRIGHT      = "PKU Alignment Team. All Rights Reserved."
PROJECT_PATH   = $(PROJECT_NAME)
SHELL          = /bin/bash
SOURCE_FOLDERS = $(PROJECT_PATH) tests docs
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -name "*.py" -o -name "*.pyi")
COMMIT_HASH    = $(shell git log -1 --format=%h)
PATH           := $(HOME)/go/bin:$(PATH)
PYTHON         ?= $(shell command -v python3 || command -v python)
PYTESTOPTS     ?=

.PHONY: default
default: install

check_pip_install = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install $(1) --upgrade)
check_pip_install_extra = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install $(2) --upgrade)

install:
	$(PYTHON) -m pip install -vvv .

install-editable:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools
	$(PYTHON) -m pip install -e .

install-e: install-editable  # alias

multi-benchmark:
	cd safepo/multi_agent && $(PYTHON) benchmark.py

single-benchmark:
	cd safepo/single_agent && $(PYTHON) benchmark.py

plot:
	cd safepo && $(PYTHON) plot.py --logdir ./runs/benchmark_multi_env
	cd safepo && $(PYTHON) plot.py --logdir ./runs/benchmark_single_env

eval:
	cd safepo && $(PYTHON) eval.py --benchmark-dir ./runs/benchmark_multi_env
	cd safepo && $(PYTHON) eval.py --benchmark-dir ./runs/benchmark_single_env

benchmark: install-editable multi-benchmark single-benchmark plot eval

pytest-install:
	$(call check_pip_install,pytest)
	$(call check_pip_install,pytest-cov)
	$(call check_pip_install,pytest-xdist)

pytest: pytest-install
	cd tests && $(PYTHON) -c 'import $(PROJECT_PATH)' && \
	$(PYTHON) -m pytest --verbose --color=yes --durations=0 \
		--cov="$(PROJECT_PATH)" --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing \
		$(PYTESTOPTS) . 