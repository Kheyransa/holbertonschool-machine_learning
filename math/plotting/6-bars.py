#!/usr/bin/env python3
"""
Defines a function that plots a stacked bar graph
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar graph representing fruit quantities per person
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    persons = ['Farrah', 'Fred', 'Felicia']
    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    width = 0.5
    bottom_val = np.zeros(3)

    for i in range(len(fruit)):
        plt.bar(
            persons,
            fruit[i],
            width=width,
            bottom=bottom_val,
            color=colors[i],
            label=fruits[i]
        )
        bottom_val += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.show()
