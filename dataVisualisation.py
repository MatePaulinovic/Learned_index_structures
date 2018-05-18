# -*- coding: utf-8 -*-
"""
@author: matep
"""

import matplotlib.pyplot as plt


def plot_loss(batch_ids, loss):
    plt.plot(batch_ids, loss)
    
    plt.xlabel('batch IDs')
    plt.ylabel('loss')
    plt.title('Training loss function')
    plt.grid(True)
    plt.savefig('loss.png')
    #plt.show()
