# Download datasets
Please download all datasets used to train pylogsentiment. After the download is finished, copy and paste the log files from various sources to directory pylogsentiment/datasets.
 
`cd pylogsentiment/datasets/` 

The directory `pylogsentiment/datasets` contains `$DATASET_NAME$/logs/` and it should look like this:

`datasets/`

`--/casper-rw/logs/`

`--/dfrws-2009-jhuisi/logs/`

`--/$DATASET_NAME$/logs/`

## Download datasets from mega.nz

There are four datasets hosted on mega.nz: `casper-rw`, `dfrws-2009-jhuisi`, `dfrws-2009-nssal`, and `honeynet-challenge-7`.

`megadl https://mega.nz/#F!yEoVHSpZ!Nj1953VllENG2uw-NgphYg`

If there is no `megadl` on your system, please install:

`sudo apt-get install megatools`

Or we can download directly on a web browser https://mega.nz/#F!yEoVHSpZ!Nj1953VllENG2uw-NgphYg

## Download datasets from LogHub

There are six datasets from LogHub and they are hosted on Zenodo. 

`wget https://zenodo.org/record/3227177/files/BGL.tar.gz`

`wget https://zenodo.org/record/3227177/files/Zookeeper.tar.gz`

`wget https://zenodo.org/record/3227177/files/Hadoop.tar.gz`

`wget https://zenodo.org/record/3227177/files/HDFS_1.tar.gz`

`wget https://zenodo.org/record/3227177/files/Thunderbird.tar.gz`

`wget https://zenodo.org/record/3227177/files/HPC.tar.gz`

## Extract all datasets

`tar -xzvf *.tar.gz`

Place all log datasets as described in the first section above with format `$DATASET_NAME$/logs/`. 