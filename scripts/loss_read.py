import zarr
import numpy as np
import matplotlib.pyplot as plt

t = zarr.open("/Users/Emile/PycharmProjects/ControlledStim/sounds/data/LossData_Emile_HighSparse/onlylast/postAnalyses_loss.zarr")

loss = np.squeeze(t)
# mean_loss = np.mean(loss, axis=0)
# plt.plot(mean_loss)
# plt.title("Mean loss over 10 trials")
# plt.show()
plt.plot(loss[0])
plt.title("Loss for the first trial")
plt.show()


