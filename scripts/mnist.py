import gzip
import sys

from data_utils import get_file
from six.moves import cPickle


def load_data(path="mnist.pkl.gz"):
    path = get_file(path, origin="https://s3.amazonaws.com/img-datasets/mnist.pkl.gz")

    if path.endswith(".gz"):
        f = gzip.open(path, "rb")
    else:
        f = open(path, "rb")

    data = cPickle.load(f, encoding="bytes")

    f.close()
    return data  # (X_train, y_train), (X_test, y_test)
