# Unifying Person and Vehicle Re-identification

Instructions for evaluating the results of our BMVC 2018 submitted paper.

The provided code was developed using the following software:
- Python 3.4
- tensorflow-gpu 1.6.0rc1
- scikit-learn 0.19.1

An MSI GeForce GTX 1070Ti and an MSI GeForce GTX 1060 OC GPU were used for training and testing the models.

# Datasets and trained models

We do not have license to include images from the individual data sets at this time. Please download them into the datasets folder as follows:

    .
    ├── ...               
    ├── datasets                
    │   ├── Cuhk03_detected          
    │   ├── Market1501
    │   ├── VehicleId
    │   └── VeRi 
    ├── excluders                
    ├── nets  
    ├── models            
    ├── test.py                
    ├── evaluate.py             
    └── ...


# Running the evaluation

Once the data sets have been downloaded and extracted, the evaluation code is ready to run. All models can be evaluated by running the `test.py` file with different arguments. We explain in detail:

- dataset: This argument represents the model that was trained on a specific dataset. The following values are available:
  - market1501
  - vehicle
  - cuhk03
  - veri
  - PVUD

- gpu: Optional argument for running the script on a specific GPU. Permitted values include every positive integer. However if no GPU is matched with the given number, the CPU will be used instead. 

- resnet_stride
  - 1 (default)
  - 2 (for TriNet)

- b4_layers
  - 1 (default)
  - 3 (for TriNet)

## Examples

We provide standard configurations for training and testing
```
python train.py \
   --experiment_root $PATH$/models/ \
   --train_dataset PVUD_train.txt \
   --image_root $PATH$/
```

```
python test.py \
--dataset PVUD \
--experiment_root $PATH$/models/MidTriNet+UT \
--image_root $PATH$/ \
--output_name test

```

# Output files

Upon termination, `test.py` will print the final results on the console and also create two files in a folder named `Results`. That folder will be located inside the corresponding dataset. For example, for the `Market-1501` dataset, the output files will be located under `./datasets/Market1501/results/`.
The files will be named according to the arguments that were given, but will always have the following endings:
- `*_evaluation.json`: This file includes the mAP metric and the CMC metrics.
- `*_ranked_images.csv`: This file contains the paths of the first 10 ranked images for every query. More specifically, it has as many rows as query images and 11 columns. The first column represents the path to the query image and the rest of the columns show the paths to each of the rank-10 images respectively.
 
# Performance discrepancies 

Please note that the calculation of the mAP score is based on the scikit-learn library and performance may deviate on other versions.

