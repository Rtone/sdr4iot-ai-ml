#!/bin/bash

.PHONY: clean data lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = sdr4iot-tensorflow-rf
PYTHON_INTERPRETER = python3
PYTHON_VENV = . venv/bin/activate; python
PIP_VENV = . venv/bin/activate; pip
VENV = . venv/bin/activate;
BUILD_DIR = build
GPULAB_CLI = gpulab-client-3.0.3.tar.gz
GPULAB_CLI_URL = https://gpulab.ilabt.imec.be/downloads/$(GPULAB_CLI)
TESTBED_CERT = cert.pem
RAW_DATASET_DIR = "$(PROJECT_DIR)/dataset"
SIGMF_URL = https://github.com/gnuradio/SigMF

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || python3 -m venv --system-site-packages ./venv
	$(VENV) pip install --upgrade pip
	$(VENV) pip install -Ur requirements.txt
	touch venv/bin/activate

test: venv
	$(VENV) nosetests test

setup_sigmf: venv
	@echo ">>> Installing SigMF package."	
	git clone $(SIGMF_URL) ||  echo ">> SigMF allready clone. Skip"
	$(VENV) cd SigMF; pip install .; cd ..;\

setup_gpulab: venv
	$(VENV) which gpulab-cli || (wget $(GPULAB_CLI_URL) && \
		pip install $(GPULAB_CLI) && rm $(GPULAB_CLI))

clean-venv:
	rm -rf venv

## Install Python Dependencies
requirements: test_environment venv setup_gpulab setup_sigmf
	pip install -r requirements.txt
	mkdir -p $(BUILD_DIR) && mkdir -p $(BUILD_DIR)/data && mkdir -p $(BUILD_DIR)/data/raw && mkdir -p $(BUILD_DIR)/html && mkdir -p $(BUILD_DIR)/data/processed && mkdir -p $(BUILD_DIR)/notebooks && mkdir -p $(BUILD_DIR)/trained_models

## Make datasets
datasets: requirements
	@echo ">> Process raw dataset from $(RAW_DATASET_DIR)"
	$(PYTHON_VENV) src/dataset_sigmf_to_csv_ble.py $(RAW_DATASET_DIR) $(BUILD_DIR)

## Make train
## use make train nb_server=(1 or 3) nn_type=(AlexNet, CNN2, ConvRNN, ResNet, CNNConv2D) exp_name=experiment name dataset_path= path to dataset (optional)
## example: make train nb_server=1 nn_type=AlexNet exp_name=$(date +%s)
train: requirements
	$(PYTHON_VENV) src/train_model.py $(nb_server) $(nn_type) $(BUILD_DIR)/$(exp_name) $(dataset_path)
    
## Make tune
## use make tune nb_server=(1 or 3) nn_type=(AlexNet, CNN2, ConvRNN, ResNet, CNNConv2D) exp_name=experiment name max_trial= max numb of trials dataset_path= path to dataset (optional)
## example: make tune nb_server=1 nn_type=AlexNet exp_name=$(date +%s) max_trial=300
tune: requirements
	$(PYTHON_VENV) src/tune_model.py $(nb_server) $(nn_type) $(BUILD_DIR)/$(exp_name) $(max_trial) $(dataset_path)

## Make notebooks
## use : make notebook notebook_name= notebook_name
## make notebook notebook_name=fingerprinting_ble_3_classes_1_server
notebook: requirements
	$(VENV) jupyter nbconvert --execute --to html --ExecutePreprocessor.timeout=-1 --output $(notebook_name).html --output-dir $(BUILD_DIR)/html notebooks/$(notebook_name).ipynb
	$(VENV) jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output $(notebook_name).ipynb --output-dir $(BUILD_DIR)/notebooks notebooks/$(notebook_name).ipynb


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf $(BUILD_DIR)

## Lint using flake8
lint:
	yapf -i -r src/

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py
	which virtualenv || pip3 install virtualenv

## Deploy jobs to GPULab
gpu_deploy: test_environment venv setup_gpulab
	$(VENV) gpulab-cli --cert $(TESTBED_CERT) submit --project sdr4iot < gpulab/train_jobDefinition.json
	$(VENV) gpulab-cli --cert $(TESTBED_CERT) submit --project sdr4iot < gpulab/tune_jobDefinition.json
	$(VENV) gpulab-cli --cert $(TESTBED_CERT) submit --project sdr4iot < gpulab/notebook_jobDefinition.json

## Deploy job to GPULab
gpu_list: test_environment venv setup_gpulab
	$(VENV) gpulab-cli --cert $(TESTBED_CERT) jobs

test_venv:
	$(VENV) which python
#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
