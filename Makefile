# ======= FOLDERS ==================
VENV := venv
PROJECT_NAME := chmncc

# ======= PROGRAMS AND FLAGS =======
PYTHON := python3
PYFLAGS := -m
PIP := pip
UPGRADE_PIP := --upgrade pip

# ======= MAIN =====================
MAIN := chmncc
MAIN_FLAGS :=
PIP := pip

# ======= DATASET ==================
DATASET:= dataset
DATASET_FLAGS :=

# ======= EXPERIMENT ===============
EXPERIMENT := experiment 
EXP_NAME := "chmncc"
EPOCHS := 10
EXPERIMENT_FLAGS := --learning-rate 0.001 --batch-size 4 --test-batch-size 4 --device cpu --project chmncc --network mlp --prediction-treshold 0.5 --fixed-confounder --dataset mnist --force-prediction --imbalance-dataset # --use-probabilistic-circuits --num-reps 2 --gates 1 --no-confound


# ======= VISUALIZE ================
VISUALIZE := visualize
VISUALIZE_FLAGS := --dataset fashion --simplified-dataset # --only-confounders true 

# ======= DEBUG ===================
DEBUG := debug
DEBUG_FLAGS := "chmncc_debug" --learning-rate 0.001 --batch-size 8 --test-batch-size 8 --device cpu --iterations 2 --network mlp --debug-folder lmao --no-integrated-gradients --force-prediction --dataset mnist --balance-subclasses X Z --balance-weights 100.0 200.0 --rrr-regularization-rate 0.001 --imbalance-dataset --fixed-confounder --correct-by-duplicating-samples  --use-probabilistic-circuits --num-reps 2 --gates 1 #--simplified-dataset #--use-softmax

# ======= ARGUMENTS ===================
ARGUMENTS := arguments
ARGUMENTS_FLAGS := --network mlp --batch-size 128 --test-batch-size 128 --device cpu --iterations 1 --arguments-folder lol --force-prediction --dataset mnist --num-element-to-analyze 300 --norm-exponent 2 --force-prediction --fixed-confounder --tau 0.1 --cincer-approach  # --use-probabilistic-circuits --num-reps 4 --gates 4 #--cincer-approach # ### # --use-gate-output # --use-gate-output

# ======= CLUSTERS ===================
CLUSTERS := clusters
CLUSTERS_FLAGS := --learning-rate 0.001 --batch-size 128 --test-batch-size 128 --device cpu --prediction-treshold 0.5 --fixed-confounder --dataset mnist --force-prediction --imbalance-dataset --use-probabilistic-circuits --num-reps 2 --gates 1 --number-of-splits 5 --number-of-epochs 2 --network mlp # --fine-tune

# ======= MULTI-STEP-ARGUMENTATION ===================
MULTI_STEP_ARGUMENTATION := multi-step-argumentation
MULTI_STEP_ARGUMENTATION_FLAGS := --network mlp --batch-size 128 --test-batch-size 128 --device cpu --iterations 1 --force-prediction --dataset mnist --norm-exponent 2 --force-prediction --fixed-confounder --tau 0.0

# ======= DOC ======================
AUTHORS := --author "Eleonora Giunchiglia, Thomas Lukasiewicz, Samuele Bortolotti" 
VERSION :=-r 0.1 
LANGUAGE := --language en
SPHINX_EXTENSIONS := --extensions sphinx.ext.autodoc --extensions sphinx.ext.napoleon --extensions sphinx.ext.viewcode --extensions myst_parser
DOC_FOLDER := docs
COPY := cp -rf
IMG_FOLDER := .github

# ======= FORMAT ===================
FORMAT := black
FORMAT_FLAG := chmncc

## Quickstart
SPHINX_QUICKSTART := sphinx-quickstart
SPHINX_QUICKSTART_FLAGS := --sep --no-batchfile --project chmncc $(AUTHORS) $(VERSION) $(LANGUAGE) $(SPHINX_EXTENSIONS)

## Build
BUILDER := html
SPHINX_BUILD := make $(BUILDER)
SPHINX_API_DOC := sphinx-apidoc
SPHINX_API_DOC_FLAGS := -P -o $(DOC_FOLDER)/source .
SPHINX_THEME = sphinx_rtd_theme
DOC_INDEX := index.html

## INDEX.rst preamble
define INDEX

.. ffdnet documentation master file, created by
   sphinx-quickstart on Fri Nov 7 23:38:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: ../../README.md
	 :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 7
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

endef

export INDEX

# ======= COLORS ===================
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
NONE := \033[0m

# ======= COMMANDS =================
ECHO := echo -e
MKDIR := mkdir -p
OPEN := xdg-open
SED := sed
TOUCH := touch
	
# RULES
.PHONY: help env install install-dev doc-layout open-doc format-code experiment dataset visualize debug arguments

help:
	@$(ECHO) '$(YELLOW)Makefile help$(NONE)'
	@$(ECHO) " \
	* env 			: generates the virtual environment using the current python version and venv\n \
	* install		: install the requirements listed in requirements.txt\n \
	* install-dev		: install the development requirements listed in requirements.dev.txt\n \
	* dataset 		: downloads and filters out the Cifar100 dataset\n \
	* experiment 		: runs the experiment\n \
	* debug 		: runs the debug of the model to solve confund\n \
	* arguments 		: runs the analyze arguments for the model to see the model confound formula behaviour\n \
	* visualize 		: shows the images associated to the specified label in the source and target dataset\n \
	* doc-layout 		: generates the Sphinx documentation layout\n \
	* doc-layout-no-theme	: generates the Sphinx documentation layout without setting the theme\n \
	* doc 			: generates the documentation (requires an existing documentation layout)\n \
	* format-code 		: formats the code employing black python formatter\n \
	* help 		: prints this message\n \
	* open-doc 		: opens the documentation\n"

env:
	@$(ECHO) '$(GREEN)Creating the virtual environment..$(NONE)'
	@$(MKDIR) $(VENV)
	@$(eval PYTHON_VERSION=$(shell $(PYTHON) --version | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]' | cut -f1,2 -d'.'))
	@$(PYTHON_VERSION) -m venv $(VENV)/$(PROJECT_NAME)
	@$(ECHO) '$(GREEN)Done$(NONE)'

install:
	@$(ECHO) '$(GREEN)Installing requirements..$(NONE)'
	@$(PYTHON) -m pip install $(UPGRADE_PIP)
	@pip install -r requirements.txt
	@$(ECHO) '$(GREEN)Done$(NONE)'

install-dev:
	@$(ECHO) '$(GREEN)Installing requirements..$(NONE)'
	@$(PYTHON) -m pip install $(UPGRADE_PIP)
	@$(PIP) install -r requirements.dev.txt
	@$(ECHO) '$(GREEN)Done$(NONE)'

experiment:
	@$(ECHO) '$(BLUE)Run the experiment..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(EXPERIMENT) $(EXP_NAME) $(EPOCHS) $(EXPERIMENT_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)'

dataset:
	@$(ECHO) '$(BLUE)Dowloading and filtering the Cifar-100 dataset..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(DATASET) $(DATASET_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)'

debug:
	@$(ECHO) '$(BLUE)Debugging the model..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(DEBUG) $(DEBUG_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)'

arguments:
	@$(ECHO) '$(BLUE)Analyzing the model arguments..$(NONE)'
	$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(ARGUMENTS) $(ARGUMENTS_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)'

multi-step-argumentation:
	@$(ECHO) '$(BLUE)Multi-step-argumentation the model arguments..$(NONE)'
	$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(MULTI_STEP_ARGUMENTATION) $(MULTI_STEP_ARGUMENTATION_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)'

visualize:
	@$(ECHO) '$(BLUE)Visualize the requested images..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(VISUALIZE) $(VISUALIZE_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)'

clusters:
	@$(ECHO) '$(BLUE)Running the cluster models..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(CLUSTERS) $(CLUSTERS_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)'

doc-layout:
	@$(ECHO) '$(BLUE)Generating the Sphinx layout..$(NONE)'
	# Sphinx quickstart
	@$(SPHINX_QUICKSTART) $(DOC_FOLDER) $(SPHINX_QUICKSTART_FLAGS)
	# Including the path for the current README.md
	@echo "import os" >> $(DOC_FOLDER)/source/conf.py
	@echo "import sys" >> $(DOC_FOLDER)/source/conf.py
	@echo "sys.path.insert(0, os.path.abspath('../..'))" >> $(DOC_FOLDER)/source/conf.py
	# Inserting custom index.rst header
	@echo "$$INDEX" > $(DOC_FOLDER)/source/index.rst
	# Sphinx theme
	@$(SED) -i -e "s/html_theme = 'alabaster'/html_theme = '$(SPHINX_THEME)'/g" $(DOC_FOLDER)/source/conf.py 
	# Add .nojekyll
	@$(TOUCH) .nojekyll
	@echo "html_extra_path = ['../../.nojekyll', '../../README.md']" >> $(DOC_FOLDER)/source/conf.py
	# Copy the image folder inside the doc folder
	@$(COPY) $(IMG_FOLDER) $(DOC_FOLDER)/source
	@$(ECHO) '$(BLUE)Done$(NONE)'

doc:
	@$(ECHO) '$(BLUE)Generating the documentation..$(NONE)'
	$(SPHINX_API_DOC) $(SPHINX_API_DOC_FLAGS)
	cd $(DOC_FOLDER); $(SPHINX_BUILD)
	@$(ECHO) '$(BLUE)Done$(NONE)'

open-doc:
	@$(ECHO) '$(BLUE)Open documentation..$(NONE)'
	$(OPEN) $(DOC_FOLDER)/build/$(BUILDER)/$(DOC_INDEX)
	@$(ECHO) '$(BLUE)Done$(NONE)'

format-code:
	@$(ECHO) '$(BLUE)Formatting the code..$(NONE)'
	@$(FORMAT) $(FORMAT_FLAG)
	@$(ECHO) '$(BLUE)Done$(NONE)'
