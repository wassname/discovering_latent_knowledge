#!/bin/bash
# Sometimes I like to do relaxed and strict requirements. The strict requirements lets you debug subtle version errors by asking "gee what exact version did they use".
# The relaxed versioning makes it easy to upgrade
set -e -x
PROJECT_NAME=dlk2
echo $PROJECT_NAME
PYTHON_INTERPRETER=~/miniforge3/envs/$PROJECT_NAME/bin/python
# minimal requirement, simpler, but no versions or pip
conda env export --no-builds --from-history > requirements/environment.min.yaml
# extensive requirements including pip and information overload
conda env export > requirements/environment.max.yaml
# requirements in a modified pip spec, usefull for dependabot and so on
$PYTHON_INTERPRETER -m pip freeze > requirements/conda.requirements.txt
# some pople like conda lock, but it doens't do pip
# cd requirements && conda-lock -f environment.max.yaml -p linux-64
