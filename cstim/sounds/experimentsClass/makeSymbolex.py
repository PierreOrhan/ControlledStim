from converting import fromDir_toDataset
from pathlib import Path

for sdir in ["word_audio","num_dd1_audio","num_dd2_audio"]:
    dir_symbol = Path("/auto/data5/speechExposureEphys/symbolex")
    fromDir_toDataset(dir_symbol/sdir,output_dir=Path(dir_symbol/("ANN_"+str(sdir))),inplace=False)
