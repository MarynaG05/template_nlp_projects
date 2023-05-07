<p align="center">
    <h1>Natural Language Processing and Information Extraction 2022WS - Project</h1>
  <img src="images/tuw_nlp.png", width="20%" height="20%" />
</p>


# Install and Quick Start

First create a new conda environment with python 3.10 and activate it:

```bash
conda create -n tuwnlpie python=3.10
conda activate tuwnlpie
```

Then install this repository as a package, the `-e` flag installs the package in editable mode, so you can make changes to the code and they will be reflected in the package.

```bash
pip install -e .
```

All the requirements are specified in the `setup.py` file with the needed versions.

## The directory structure and the architecture of the project is the following:

```
📦project-2022WS
 ┣ 📂data
 ┃ ┣ 📜README.md
 ┃ ┣ 📜bayes_model.tsv
 ┃ ┣ 📜bow_model.pt
 ┃ ┗ 📜imdb_dataset_sample.csv
 ┣ 📂docs
 ┃ ┗ 📜milestone1.ipynb
 ┣ 📂images
 ┃ ┗ 📜tuw_nlp.png
 ┣ 📂scripts
 ┃ ┣ 📜evaluate.py
 ┃ ┣ 📜predict.py
 ┃ ┗ 📜train.py
 ┣ 📂tests
 ┃ ┣ 📜test_milestone1.py
 ┃ ┣ 📜test_milestone2.py
 ┣ 📂tuwnlpie
 ┃ ┣ 📂milestone1
 ┃ ┃ ┣ 📜model.py
 ┃ ┃ ┗ 📜utils.py
 ┃ ┣ 📂milestone2
 ┃ ┃ ┣ 📜model.py
 ┃ ┃ ┗ 📜utils.py
 ┃ ┗ 📜__init__.py
 ┣ 📜.gitignore
 ┣ 📜LICENSE
 ┣ 📜README.md
 ┣ 📜setup.py
```

- `data`: This folder contains the data that you will use for training and testing your models. You can also store your trained models in this folder. The best practice is to store the data elsewhere (e.g. on a cloud storage) and provivde download links. If your data is small enough you can also store it in the repository.
- `docs`: This folder contains the reports of your project. You will be asked to write your reports here in Jupyter Notebooks or in simple Markdown files.
- `images`: This folder contains the images that you will use in your reports.
- `scripts`: This folder contains the scripts that you will use to train, evaluate and test your models. You can also use these scripts to evaluate your models.
- `tests`: This folder contains the unit tests for your code. You can use these tests to check if your code is working correctly.
- `tuwnlpie`: This folder contains the code of your project. This is a python package that is installed in the conda environment that you created. You can use this package to import your code in your scripts and in your notebooks. The `setup.py` file contains all the information about the installation of this repositorz. The structure of this folder is the following:
  - `milestone1`: This folder contains the code for the first milestone. You can use this folder to store your code for the first milestone.
  - `milestone2`: This folder contains the code for the second milestone. You can use this folder to store your code for the second milestone.
  - `__init__.py`: This file is used to initialize the `tuwnlpie` package. You can use this file to import your code in your scripts and in your notebooks.
- `setup.py`: This file contains all the information about the installation of this repository. You can use this file to install this repository as a package in your conda environment.
- `LICENSE`: This file contains the license of this repository.
- `team.cfg`: This file contains the information about your team.


# Running the code

## Run Milestone 1

## Training and Evaluation:

To train a model on the SEMEVAL dataset and then save the weights to a file, you can run a command:

```bash
scripts/train.py milestone-1
```
```bash
scripts/predict.py milestone-1
```
```bash
scripts/evaluate.py milestone-1
```
## Reflexion and Findings:

Naive Bayes is a popular choice for text classification tasks in natural language processing (NLP) because it is a simple yet effective probabilistic model that can be trained quickly with relatively small datasets.

The simplicity, speed, and efficiency of Naive Bayes make it a viable option for a baseline model in NLP projects since it handles text categorization tasks well. To find the optimal model for your particular issue, it is usually a good practice to examine and compare several models. More sophisticated models could, however, perform better for particular datasets or activities.

Our baseline model scores an accuracy of 60.51 %, which is decent. The dataset we used for training has 8000 entries, which are categorized in 10 different relation types. One could say that we don’t encounter high dimensional data and therefor the model doesn’t perform as well as it could. To test that assumption, we took subset with 5, and 7 different relation types and checked how they performed.

With 7 different relation types of the model scored a accuracy of 47.77%, if we reduced the dimension to only 5 different relation types the model scores an accuracy of 36.40%.

We can conclude that picking a naïve Bayes classifier for our baseline model, was the right choice, as it helped us getting familiar with the topic and the prediction for our data set were reasonably. But there are some down sights to it, and we are keen to see how the model we pick for milestone 2 will perform compared to the baseline model.

## Run Milestone 2

## Training and Evaluation:

To train a model with a training loop on the SEMEVAL dataset and then save the weights to a file, you can run a command:

```bash
scripts/train.py milestone-2
```
To use the model to make predictions on the SEMEVAL dataset and then save the predictions to a file, you can run a command:

```bash
scripts/predict.py milestone-2
```

## Reflexion and Findings:

For a neural network implementation we chose the Bow Classifier with an embedding layer, the job of this layer is to take the words into our network and return a dense vector representation for each word. We tried tuning the hyperparameters, for example the one we experimented with the most is the embedding layer and its dimension size. We implemented a confusion matrix inside of the prediction script which gives us more insight into the classification errors for each and every label. We experimented with different activation functions such as softmax and log softmax, we also tried using different types of pooling layers such as: average, max, min and for now it seems that the average pooling layer gives the best results. 

We tested different number of epochs, where we tried to have the model not overfit for the data and the optimal number of epochs we found for now is 10. The precision value after 10 epochs for the training set is 78% and for the validation set it is 58%. 
