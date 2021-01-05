# Taxifare Deep

This repository is a python-package that allows to fit a deep learning model predicting the price of a taxi course using the [TaxiFare](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) dataset.

## How to test it directly from Google Collab ?

1. Open [colab.research.google.com](https://colab.research.google.com/)
2. Create a new collab notebook, and change the runtime to GPU
3. Pip-install this repo (which is a package) directly from within your collab notebook, by running this cell below

```python
! pip install --quiet git+https://github.com/brunolajoie/taxifare_deep
```

## Basic workflow

You can now try to fit and predict using the following basic workflow

```python
# Import Trainer class
from taxifare_deep.trainer import Trainer

# Download a sub-sample of rows to train on
trainer = Trainer(nrows=10000)

# Clean data
trainer.clean()

# Preprocess data and create train/test/split
trainer.preproc(test_size=0.3)

# Fit neural network and show training performance
trainer.fit(plot_history=True, verbose=1)

# evaluate on test set (by default the holdout from train/test/split)
trainer.evaluate(X_test=None, y_test=None)
```

## Investigate code base

- Feel free to fork this repo to your own github account, and clone it to your local hard drive
- Try to understand how the code works, following basic workflow above (you can run it on a local notebook after doing `pip install -e .` from the root folder)
- Feel free to improve it as you see fit


