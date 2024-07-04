# Installation  
Step 1: Create a new conda environment:
```
conda create -n owl python=3.9
conda activate owl
```
Step 2: Install relevant packages
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install transformers==4.28.0 datasets==2.17.0 wandb sentencepiece
pip install accelerate==0.18.0
```
