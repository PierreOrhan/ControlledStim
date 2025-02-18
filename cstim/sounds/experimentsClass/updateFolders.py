import pandas as pd
from pathlib import Path
import os
import numpy as np
def updateFolders(old_folder :str,new_folder : str):
    # In an ANN dataset, update the .wavs path link and sound info and mask info as stored
    # in the trials.csv
    # Useful for uploading a dataset on the hub and then redownloading it locally
    # or sending a datatset on a cluster

    df = pd.read_csv(Path(new_folder)/"trials.csv")
    wav_paths,sip = [],[]
    for id in range(len(df)):
        wav_paths += [df.iloc[id]["wav_path"].replace(old_folder,new_folder)]
        sip+=[df.iloc[id]["sound_info_path"].replace(old_folder, new_folder)]

    new_df = pd.DataFrame({"wav_path":np.array(wav_paths),"sound_info_path":np.array(sip)})
    new_df.to_csv(Path(new_folder)/"trials.csv")

old_folder = "/media/pierre/NeuroData2/datasets/TCI_ephys2/sub-frinault"
new_folder = "/gpfsscratch/rech/fqt/uzz43va/NeuroData/TCI_ephys/stimulis/sub-frinault"
for sess in os.listdir(new_folder):
    updateFolders(old_folder+"/"+sess,new_folder+"/"+sess)