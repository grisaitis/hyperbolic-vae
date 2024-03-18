curl https://raw.githubusercontent.com/GitAlias/gitalias/master/gitalias.txt -o ~/.gitalias
git config --global include.path ~/.gitalias
PIP_NO_DEPS=1 conda env create -f conda-env-cpu.yml -y