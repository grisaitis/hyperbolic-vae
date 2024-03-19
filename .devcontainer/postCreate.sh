curl https://raw.githubusercontent.com/GitAlias/gitalias/master/gitalias.txt -o ~/.gitalias
git config --global include.path ~/.gitalias
python -m pip install --upgrade pip
pip install poetry
poetry config virtualenvs.in-project true
poetry install --no-interaction