import numpy as np
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt

print(matplotlib.get_backend())

random_image = np.random.random([500, 500])
print(random_image)
plt.imshow(random_image, cmap='gray')
plt.colorbar()
plt.show()

