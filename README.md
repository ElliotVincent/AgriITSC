<div align="center">
<h2>
Pixel-wise Agricultural Image Time Series Classification: Comparisons and a Deformable Prototype-based Approach<p></p>

<a href="https://imagine.enpc.fr/~vincente/">Elliot Vincent</a>&emsp;
<a href="https://www.di.ens.fr/~ponce/">Jean Ponce</a>&emsp;
<a href="https://imagine.enpc.fr/~aubrym/">Mathieu Aubry</a>

<p></p>

</h2>
</div>

Official PyTorch implementation of [**Pixel-wise Agricultural Image Time Series Classification: Comparisons and a Deformable Prototype-based Approach**](https://imagine.enpc.fr/~vincente/).
Check out our [**webpage**](https://imagine.enpc.fr/~vincente/) for other details!

![alt text](https://github.com/ElliotVincent/AgriITSC/blob/main/agriitsc_teaser.png?raw=true)

If you find this code useful, don't forget to <b>star the repo :star:</b>.


## Installation :construction_worker:

### 1. Clone the repository in recursive mode

```
git clone git@github.com:ElliotVincent/AgriITSC.git --recursive
```

### 2. Create and activate virtual environment

```
python3 -m venv agriitsc
source agriitsc/bin/activate
python3 -m pip install -r requirements.txt
```
This implementation uses Pytorch.
## How to use :rocket:

We present steps to run our method on TimeSen2Crop, making our pre-pocessed version of this dataset available here:

- :austria: [**TimeSen2Crop**](https://drive.google.com/file/d/1rCIyB4LETzfBhfYoc7dLHNKYhv8vJ315/) [2]

To train and evaluate on other datasets, please follow the links below. 
All information on how we process the data is described in our paper.

- :fr: [**PASTIS**](https://zenodo.org/record/5012942) [1]
- :south_africa: [**SA**](https://mlhub.earth/data/ref_fusion_competition_south_africa) [4]
- :de: [**DENETHOR**](https://mlhub.earth/data/dlr_fusion_competition_germany) [3]
 
### 1. Downlaoad the dataset

```bash
cd AgriITSC
mkdir datasets && cd datasets
gdown --id 1rCIyB4LETzfBhfYoc7dLHNKYhv8vJ315
unzip TimeSen2Crop.zip
```

### 2. Training the model

To train and evaluate our method with supervision do:

```bash
PYTHONPATH=$PYTHONPATH:./src python3 src/trainer.py -t supervised -c ts2c_sup.yaml
```

And without supervision do:

```bash
PYTHONPATH=$PYTHONPATH:./src python3 src/trainer.py -t unsupervised -c ts2c_unsup.yaml
```

### 3. Saved model

Our trained models on TimeSen2Crop are available in `results/`, both for the supervised and unsupervised case.

## Bibliography

[1] Vivien Sainte Fare Garnot et al. Panoptic segmentation of satellite image time series with convolutional temporal attention networks. ICCV, 2021.

[2] Giulio Weikmann et al. Timesen2crop: A million labeled samples dataset of sentinel 2 image time series for crop-type classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2021.

[3] Lukas Kondmann et al. Denethor: The dynamicearthnet dataset for harmonized, inter-operable, analysis-ready, daily crop monitoring from space. NeurIPS Datasets and Benchmarks Track, 2021.

[4] Lukas Kondmann et al. Early crop type classification with satellite imagery: an empirical analysis. ICLR 3rd Workshop on Practical Machine Learning in Developing Countries, 2022.

