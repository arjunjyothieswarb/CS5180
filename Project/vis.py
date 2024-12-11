import numpy as np
import matplotlib.pyplot as plt

a = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
              [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
              [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
              [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
              [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
              [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
              [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
              [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

b = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])

plt.imshow(a, cmap='Greys')
plt.axis('off')
plt.show()