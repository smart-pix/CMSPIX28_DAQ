# CMSPIX28_DAQ
DAQ for CMSPIX28 ASIC

# To setup with all modules and submodules on master
Might require adding ssh private/public key pair before cloning repo. Example instructions: [stackoverflow](https://stackoverflow.com/questions/2643502/how-to-solve-permission-denied-publickey-error-when-using-git).
```
git clone --recursive git@github.com:smart-pix/CMSPIX28_DAQ.git
git submodule foreach --recursive 'if git show-ref --verify --quiet refs/heads/main; then git checkout main; elif git show-ref --verify --quiet refs/heads/master; then git checkout master; else echo "Neither main nor master branch exists"; fi'
git submodule foreach --recursive git pull
```

# spacely-asic-config
The following copies an instance of spacely-asic-config files inside PySpacely:
```
chmod +x setup.sh
./setup.sh
```

# Python environment
The python environment in .../CMS_PIX_28/testing/tools/venvs/p3.11.11/bin/ contains the right modules installed to run spacely.

# Running only-FPGA setup
- If in case, you need to run spacely **without the carboard**, then comment out [these](https://github.com/SpacelyProject/spacely/blob/f5421e79060af1a8675ec8a453c58c70b562c728/PySpacely/src/Spacely_Caribou.py#L255-L256) lines.
