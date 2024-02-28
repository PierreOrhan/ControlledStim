from pathlib import Path
from sounds.perExperiment.protocols.Pena2002 import AXCSyllableStream_exp1


output_dir = Path("/media/pierre/Crucial X8/datasets/BoubenecLab/debugDatasets") / "Pena2002_16000"
protocol = AXCSyllableStream_exp1(samplerate = 16000)
protocol.generate(n_trial=10,output_dir=output_dir)