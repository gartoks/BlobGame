If using a conda environment, the following command is needed:
```sh
conda install -c conda-forge gcc=12.1.0
```

For a cpu only installation use
```sh
pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Tensorboard can be started using this command in the project root.
```sh
tensorboard --logdir=. --host localhost --port 8888
```

The game needs these arguments:
```sh
BlobGame --sockets true 123649 1337 Classic
```

Currently only `LearnDWN_pytorch.py` is tested. The other ones shouldn't be to hard to get working.