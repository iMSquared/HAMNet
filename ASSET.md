## Assets Setup

We release a pre-processed version of the object mesh assets from [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) in [here](https://huggingface.co/imm-unicorn/corn-public/resolve/main/DGN.tar.gz).

After downloading the assets, extract them to `/path/to/data/DGN` in the _host_ container, so that `/path/to/data` matches the directory
configured in [docker/run.sh](docker/run.sh), i.e.

```bash
mkdir -p /path/to/data/DGN
tar -xzf DGN.tar.gz -C /path/to/data/DGN
```

Or you could run the provided script inside the docker container to automatically download the DGN dataset:
```bash
bash download.sh --dgn
```

so that the resulting directory structure _inside_ the docker container looks as follows:

```bash
$ tree /input/DGN --filelimit 16 -d     

/input/DGN
|-- coacd
`-- meta-v8
    |-- cloud
    |-- cloud-2048
    |-- code
    |-- hull
    |-- meta
    |-- new_pose
    |-- normal
    |-- normal-2048
    |-- pose
    |-- unique_dgn_poses
    `-- urdf
```


**Download Training/Evaluation Episodes**

For downloading episodes for both training and evaluation, as well as misc files for setup, 
please run the script with:
```bash
bash download.sh
```
This will download all necessary episodes (eps-demo.pth, eps-eval.pth) and setup files.
