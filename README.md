# Taxifare Deep

Basic Deep Learning model applied to the TaxiFare problem.

## Test it with Google Collab

Open `notebook/taxifare_deep.ipynb` with collab and run

```python
! pip install --quiet git+https://github.com/brunolajoie/taxifare_deep
```

## Basic workflow

```python
from taxifare_deep.trainer import Trainer

# Instanciate trainer with number of rows to download and use
trainer = Trainer(nrows=10000)

# clean data
trainer.clean()

# Preprocess data and create train/test/split
trainer.preproc(test_size=0.3)

# Fit neural network and show training performance
trainer.fit(plot_history=True, verbose=1)

# evaluate on test set (by default the holdout from train/test/split)
trainer.evaluate(X_test=None, y_test=None)
```
