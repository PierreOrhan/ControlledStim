Beyond the generation of the waveform, one need to control several very important parameters:
sampling rate, normalization ,...

Sampling_rate:
Downstream models are trained with a certain sampling rate,
it is therefore very important to generate sounds according to their target sampling rate, 
such that one controls any distortion that could happen due to resampling.

Normalization:
One need to be very careful with normalization. For example if one were to normalize using the standard deviation of the signal,
silences in the signal would prevent this normalization from providing the good normalization level.

There are two ways to easy way to normalize an audio waveform: peak normalization or RMS normalization.
TODO: definition of the first, then definition of the second.

We make sure every generated signal is peaked-normalized such that its maximal value is 1.
Protocols often do not indicate which normalization they have been using.