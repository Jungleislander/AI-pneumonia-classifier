# Variables
ENV_NAME = ai-pneumonia-classifier          	 # Name of the conda environment
PYTHON_VERSION = 3.12.2      	# Default Python version if environment.yml is not present
ENV_FILE = environment.yml   		# Name of the environment file

# Targets

.PHONY: create update conda-env clean activate deactivate notebook freeze verify

create: conda-env

# Create a conda environment from environment.yml
conda-env:
	@echo "Creating conda environment $(ENV_NAME) from $(ENV_FILE)..."
	conda env create  --name $(ENV_NAME) --file $(ENV_FILE)

# Update a conda environment from environment.yml
update:
	@echo "Updating conda environment $(ENV_NAME) from $(ENV_FILE)..."
	conda env update -n $(ENV_NAME) --file $(ENV_FILE)

# Clean up the conda environment
clean:
	@echo "Removing conda environment $(ENV_NAME)..."
	conda remove --name $(ENV_NAME) --all -y

# Step to activate the environment in the current shell session
activate:
	@echo "Unable to activate from the Makefile so in order to activate the environment, run:"
	@echo "conda activate $(ENV_NAME)"

deactivate:
	conda deactivate

notebook:
	jupyter notebook  

# Export the environment to environment.yml
freeze:
	@echo "Exporting conda environment $(ENV_NAME) to $(ENV_FILE)..."
	conda env export --no-builds | tail -r | tail -n +2 | tail -r > environment.yml

verify:
	@echo "Ensure you are in the correct conda environment. There should be an * next to the name of the current active env."
	conda info --envs

run:
	python app.py