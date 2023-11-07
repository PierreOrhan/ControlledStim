import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


# Constants
sample_rate = 44100  # 44.1 kHz
duration_N = 1  # 1 second
duration_RN = 0.5  # 0.5 seconds

# Generate N stimulus
n_samples_N = int(sample_rate * duration_N)
noise_N = np.random.normal(0, 1, n_samples_N)  # Gaussian noise with mean 0 and std deviation 1

# Generate RN stimulus by concatenating two identical N stimuli
n_samples_RN = int(sample_rate * duration_RN)
noise_RN = np.random.normal(0,1,n_samples_RN)
noise_RN = np.append(noise_RN, noise_RN)


# Applying hamming window: necessary??
H = np.hamming(len(noise_N))
noise_RN = noise_RN*H
noise_N = noise_N*H

# Save N and RN stimuli as audio files

sf.write("/Users/Emile/Documents/Polytechnique/Cours/Projet3A/N_stimulus.wav", noise_N, sample_rate)
sf.write("/Users/Emile/Documents/Polytechnique/Cours/Projet3A/RN_stimulus.wav", noise_RN, sample_rate)


fig, (ax1, ax2) = plt.subplots(1,2, sharey = True)
ax1.specgram(noise_N)
ax2.specgram(noise_RN)
plt.show()

