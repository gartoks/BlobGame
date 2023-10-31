If using a conda environment, the following command is needed:
```sh
conda install -c conda-forge gcc=12.1.0
```

for a cpu only installation use
```sh
pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```