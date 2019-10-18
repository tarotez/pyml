# -*- coding: utf-8 -*-
import matplotlib.pyplot as plot
from mnist import load_data
train_set, test_set = load_data()
train_set_x, train_set_y = train_set
i = 5
plot.imshow(train_set_x[i], cmap='Greys_r')
