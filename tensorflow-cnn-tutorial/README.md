For installation, created a virtual env and ran:

    pip install tensorflow
    pip install matplotlib
    pip install tensorflow[and-cuda]

Then, run `train.py`, it saves weights to a `.keras` file (which is not in Git).
Then `predict.py` loads the weights from this file and runs predictions.
