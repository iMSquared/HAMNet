## Installation

### Isaac Gym

First, download isaac gym from [here](https://developer.nvidia.com/isaac-gym) and extract them to the `${IG_PATH}` host directory
that you configured during docker setup. It maps to`/opt/isaacgym` directory inside the container.
In other words, the resulting directory structure should look like (in container):

```bash
$ tree /opt/isaacgym -d -L 1

/opt/isaacgym
|-- assets
|-- docker
|-- docs
|-- licenses
`-- python
```

(If `tree` command is not found, you may simply install it via `sudo apt-get install tree`.)

Afterward, follow the instructions in the referenced page to install the isaac gym package.

Alternatively, assuming that the isaac gym package has been downloaded and extracted in the correct directory(`/opt/isaacgym`),
we provide the default setups for isaac gym installation in the [setup script](./setup.sh)
in the [following section](#python-package), which handles the installation automatically.

### Python Package

Then, inside the docker container, run the [setup script](./setup.sh):

```bash
bash setup.sh
```


To test if the installation succeeded, you can run:
```bash
python3 -c 'import isaacgym; print("OK")'
python3 -c 'import ham; print("OK")'
```
