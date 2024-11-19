# store top directory
top=${PWD}

# clone submodules
git submodule update --init

# move each submodule to master/main and pull
git submodule foreach 'if git show-ref --verify --quiet refs/heads/main; then git checkout main; elif git show-ref --verify --quiet refs/heads/master; then git checkout master; else echo "Neither main nor master branch exists"; fi'
git submodule foreach git pull

# setup spacely for testing and get smart pix routines
cd spacely/PySpacely/
mkdir spacely-asic-config
cd spacely-asic-config
git clone git@github.com:smart-pix/CMSPIX28Spacely.git

# clean up and go back to main dir
cd $top
