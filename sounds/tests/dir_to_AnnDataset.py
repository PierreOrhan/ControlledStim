import os
import shutil
import pandas as pd
import os
import numpy as np
from pathlib import Path
from typing import Union,Optional
# import soundfile as sf
import scipy.io
import tqdm
import julius
import torchaudio
from pathlib import  Path


def fromDir_toDataset(input_dir : Union[Path,str],output_dir : Optional[Union[Path,str]] = None,inplace:bool =True,
                      new_sr = 16000) -> None:
    """
    Simply transform a directory of files into a novel directory, organised as a dataset generated by ControlledStim
    :param input_dir: Path or str, the directory with wav files.
    :param output_dir: Path or str, the output directory, structured as a ControlledStim dataset.
    :param inplace: if we perform the generation of the dataset in-place.
    :return:
    """
    if output_dir==None:
        assert inplace , 'if no output_dir, inplace should be True'
    names = os.listdir(input_dir)
    assert np.all([n.endswith(".wav") for n in names])
    if inplace:
        output_dir=input_dir
    os.makedirs(output_dir/"sounds",exist_ok=True)
    durations = []
    sound_info_paths = []

    ## Generate the resampler:
    sd,srorig = torchaudio.load(input_dir/names[0])
    resampler = julius.ResampleFrac(old_sr= srorig,new_sr=new_sr)
    os.makedirs(output_dir / "sound_info", exist_ok=True)
    os.makedirs(output_dir / "sounds", exist_ok=True)
    for name in tqdm.tqdm(names):
        df_sound_info = pd.DataFrame()
        df_sound_info["name"] = [name]
        sound_info_paths += [str(Path(output_dir) / "sound_info" / name.replace(".wav",".csv"))]
        if inplace:
            if input_dir/name != output_dir/"sounds"/name:
                shutil.move(input_dir/name,output_dir/"sounds"/name)
        else:
            if input_dir/name != output_dir/"sounds"/name:
                shutil.copy(input_dir/name,output_dir/"sounds"/name)
        sd,sr = torchaudio.load(input_dir / name)
        durations += [sd.shape[-1]/sr]
        df_sound_info["duration"] = [durations[-1]]
        if sr!=srorig:
            resampler = julius.ResampleFrac(old_sr=sr, new_sr=new_sr) # change resampler
            srorig = sr
        new_sd = resampler(sd)
        os.remove(output_dir/"sounds"/name)

        scipy.io.wavfile.write(filename=str(output_dir/"sounds"/name),data=new_sd.numpy()[0,:],rate=new_sr)
        df_sound_info.to_csv(Path(output_dir) / "sound_info" / (name.replace(".wav",".csv")), index=False)

    df = pd.DataFrame()
    df["name"] = names
    df["wav_path"] = [str(Path(output_dir)/"sounds"/n) for n in names]
    df["duration"] = durations
    df["sound_info_path"] = sound_info_paths
    df.to_csv(Path(output_dir) / "trials.csv", index=False)


# input_dir = Path("/auto/data5/speechExposureEphys/oscipek/nsd/sounds")
# output_dir = Path("/media/pierre/NeuroData2/datasets/speechExposure_ltq_nsd/nsd")
# fromDir_toDataset(input_dir,output_dir,inplace=False)
#
# input_dir = Path("/auto/data5/speechExposureEphys/oscipek/ltq/sounds")
# output_dir = Path("/media/pierre/NeuroData2/datasets/speechExposure_ltq_nsd/ltq")
# fromDir_toDataset(input_dir,output_dir,inplace=False)

# input_dir = Path("/media/pierre/NeuroData2/datasets/NSD_fmri_paradigm1/stimulis/sounds")
# output_dir = Path("/media/pierre/NeuroData2/datasets/NSD_fmri_paradigm1/stimulis")
# fromDir_toDataset(input_dir,output_dir,inplace=False)
#
# input_dir = Path("/media/pierre/NeuroData2/datasets/syntaxicProbingVseamless/modelmatch/generated/")
# output_dir = Path("/media/pierre/NeuroData2/datasets/syntaxicProbingVseamless/modelmatch/stimulis/")
input_dir = Path("/gpfsscratch/rech/fqt/uzz43va/NeuroData/syntaxicProbingVseamless/modelmatch/generated/generated")
output_dir = Path("/gpfsscratch/rech/fqt/uzz43va/NeuroData/syntaxicProbingVseamless/modelmatch/stimulis/")
for e in os.listdir(input_dir):
    fromDir_toDataset(input_dir/e,output_dir/e,inplace=False)