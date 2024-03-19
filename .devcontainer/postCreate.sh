curl https://raw.githubusercontent.com/GitAlias/gitalias/master/gitalias.txt -o ~/.gitalias
git config --global include.path ~/.gitalias
# conda env create -f .devcontainer/conda-env-cpu.yml -y
# conda env update --name base --file .devcontainer/conda-env-cpu.yml -y
# conda activate hyperbolic-vae
# pip install --no-deps -e .
conda install -c conda-forge python=3.11 poetry -y
# python -m pip install poetry
pip install -e .