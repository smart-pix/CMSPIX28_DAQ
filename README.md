# CMSPIX28_DAQ
DAQ for CMSPIX28 ASIC

# To setup with all modules and submodules on master
```
git clone --recursive git@github.com:badeaa3/CMSPIX28_DAQ.git
git submodule foreach --recursive 'if git show-ref --verify --quiet refs/heads/main; then git checkout main; elif git show-ref --verify --quiet refs/heads/master; then git checkout master; else echo "Neither main nor master branch exists"; fi'
git submodule foreach --recursive git pull
```