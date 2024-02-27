import zarr
import numpy as np
import matplotlib.pyplot as plt

t = zarr.open("/Users/Emile/PycharmProjects/ControlledStim/sounds/data/LossData_Emile_Sparse/onlylast/postAnalyses_loss.zarr")

loss = np.squeeze(t)
print(loss)
mean_loss = np.mean(loss, axis=0)


plt.plot(mean_loss)
plt.show()
plt.plot(loss[0])
plt.show()


