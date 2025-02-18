import julius
import librosa
from pathlib import Path

import scipy.io.wavfile
import torch
import scipy.signal as sc
import sounddevice as sde
import numpy as np
from TCI.signals.enveloppe import generate_wideband_gamma

dir = Path("/auto/data5/speechExposureEphys/Vocoder/stim502_french_orig.wav")

dir = Path("/auto/data5/speechExposureEphys/Vocoder/sound_defense.wav")

sd,sr = librosa.load(dir,sr=None)

# cuttofs= [50,229,558,1161,2265,4290,8000]
cuttofs = np.round(np.logspace(np.log(50)/np.log(10),np.log(8000)/np.log(10),num=4),0)
bps = [julius.filters.BandPassFilter(cutoff_low=c/sr,cutoff_high=c2/sr,zeros=4) for c,c2 in zip(cuttofs[:-1],cuttofs[1:])]
sd_out = torch.stack([bp(torch.tensor(sd)) for bp in bps])

amplitude_env = torch.abs(sd_out)
M = int(0.064*sr)
kaiser_window = sc.windows.kaiser(M,beta=20,sym=True)
filtered_env = torch.conv1d(
                            torch.nn.functional.pad(amplitude_env,(M//2,M//2-1),mode="constant")[:,None,:],
                        torch.tensor(kaiser_window[::-1].copy(),dtype=torch.float32)[None,None,:])[:,0,:]
# noise = torch.stack([torch.tensor(generate_wideband_gamma(sr,sd.shape[0]/sr,lowcut=c,highcut=c2,numtaps=512))
#                      for c,c2 in zip(cuttofs[:-1],cuttofs[1:])])

noise = torch.tensor(np.random.normal(0,1,sd_out.shape[-1]),dtype=torch.float32)
noise = torch.stack([bp(torch.tensor(noise)) for bp in bps])
vocoded_sound = torch.sum(filtered_env*noise,dim=0)
vocoded_sound = vocoded_sound*torch.max(amplitude_env)/torch.max(torch.abs(vocoded_sound))

cat_sound = torch.cat([vocoded_sound,torch.tensor(sd),vocoded_sound],dim=-1)
sde.play(cat_sound,samplerate=sr)
import scipy.io.wavfile
scipy.io.wavfile.write("/auto/data5/speechExposureEphys/Vocoder/defense_vocoded.wav",data=cat_sound.numpy(),rate=sr)

sde.play(vocoded_sound,samplerate=sr)
sde.play(sd,samplerate=sr)
sde.play(vocoded_sound,samplerate=sr)


fig,ax = plt.subplots()
ax.plot(vocoded_sound)
ax.plot(sd)
ax.set_xlim(60000,75000)
fig.show()


import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax.matshow(sd_out)
ax.set_aspect(sd_out.shape[1]/sd_out.shape[0])
fig.show()

# The words were first filtered into six
# logarithmically spaced frequency bands between 50 and 8000 Hz.
# Contiguous band-pass filters were constructed in the frequency
# domain: Passbands were 3 dB down at 50, 229, 558, 1161, 2265,
# 4290, and 8000 Hz with a roll-off of 22 dB per octave (cutoff
# frequencies chosen to simulate equal distances along the basilar
# membrane; Greenwood, 1990). For each spoken word, the ampli-
# tude envelopes of the energy contained within each frequency
# band were extracted via the standard Praat algorithm (squaring
# intensity values and convolving with a 64-ms Kaiser-20 window,
# removing pitch-synchronous oscillations above 50 Hz). The result-
# ing envelopes were then applied to band-pass filtered noise in the
# same frequency ranges. Finally, the resulting bands of modulated
# noise were recombined to produce the distorted word.