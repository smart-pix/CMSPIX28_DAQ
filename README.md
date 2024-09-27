# CMSPIX28_DAQ
DAQ for CMSPIX28 ASIC

# To setup with all modules and submodules on master
```
git clone --recursive git@github.com:smart-pix/CMSPIX28_DAQ.git
git submodule foreach --recursive 'if git show-ref --verify --quiet refs/heads/main; then git checkout main; elif git show-ref --verify --quiet refs/heads/master; then git checkout master; else echo "Neither main nor master branch exists"; fi'
git submodule foreach --recursive git pull
```

# Notes to get running
- Need to comment out the following lines https://github.com/SpacelyProject/spacely/blob/main/PySpacely/src/Spacely_Caribou.py#L100-L101. In our peary device file we do not have the car_i2c_write because we do not have a car board yet.