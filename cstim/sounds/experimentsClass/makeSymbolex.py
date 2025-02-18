from converting import fromDir_toDataset
from pathlib import Path

dir_word = Path("/auto/data5/speechExposureEphys/symbolex/num_dd2_audio")
fromDir_toDataset(dir_word,output_dir=Path(str(dir_word).replace("num_dd2_audio","ANN_num_dd2_audio")),inplace=False)
