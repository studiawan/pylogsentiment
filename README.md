# pylogsentiment
We name the proposed method as `pylogsentiment`.

## How to install
To run the `pylogsentiment` tool, please follow these steps.

1. Clone the repository

   `git clone https://github.com/studiawan/pylogsentiment.git`

2. Change directory to `pylogsentiment`

   `cd pylogsentiment`

3. Create virtual environment using anaconda:

   `conda create --name pylogsentiment python=3.5`
   
   and then activate it:
    
    `conda activate pylogsentiment`
   
   If you do not have anaconda, use any other tools to create virtual environment. We highly recommend to install `pylogsentiment` on a virtual environment.

5. Install `pylogsentiment`

   `pip install -e .`
   
## Download the model file

To run `pylogsentiment`, we need to download the model file and word index file. When downloading the datasets using `megadl` command, both files are also downloaded. Please [read here](https://github.com/studiawan/pylogsentiment/blob/master/datasets/README.md) for instructions.
Note that both files should be placed in `pylogsentiment/datasets/` directory.

## How to run `pylogsentiment`
To run `pylogsentiment`, type the command:

`python pylogsentiment/pylogsentiment.py -i log_file.log -o results_file.csv`

where `log_file.log` is the input log file and `results_file.csv` is the anomaly detection results in a CSV file.

## Downloading the datasets

Follow the instructions here: [Download the datasets](https://github.com/studiawan/pylogsentiment/blob/master/datasets/README.md)

## Building the ground truth

If you want to build the ground truth by your own, follow these steps. In the project root directory, run script `groundtruth.py` followed by dataset name. For example, the dataset names are `casper-rw, dfrws-2009-jhuisi, dfrws-2009-nssal, honeynet-challenge7`. For example:

   `python pylogsentiment/groundtruth/groundtruth.py casper-rw`

## Training your own model with `pylogsentiment`

To train your own model, please download and build ground truth as described above. Subsequently, download and extract GloVe word embedding as [described here](https://github.com/studiawan/pylogsentiment/blob/master/glove/README.md). Then, we can run this command:

`python pylogsentiment/experiment/experiment.py pylogsentiment`

The final model is located in directory `datasets/best-model-pylogsentiment.hdf5`
