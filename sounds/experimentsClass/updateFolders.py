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
    wav_paths,sip,mip = [],[],[]
    add_mip = False
    add_sip = False
    for id in range(len(df)):
        wav_paths += [df.iloc[id]["wav_path"].replace(old_folder,new_folder)]
        if "sound_info_path" in df.iloc[id].keys():
            add_sip = True
            sip+=[df.iloc[id]["sound_info_path"].replace(old_folder, new_folder)]
        if "mask_info_path" in df.iloc[id].keys():
            add_mip = True
            mip += [df.iloc[id]["mask_info_path"].replace(old_folder, new_folder)]
    if add_mip and add_sip:
        new_df = pd.DataFrame({"wav_path":np.array(wav_paths),"sound_info_path":np.array(sip),
                               "mask_info_path":np.array(mip)})
    elif add_sip:
        new_df = pd.DataFrame({"wav_path": np.array(wav_paths), "sound_info_path": np.array(sip)})
    else:
        new_df = pd.DataFrame({"wav_path": np.array(wav_paths)})

    df.pop("wav_path")
    try:
        df.pop("sound_info_path")
    except:
        pass
    if add_mip:
        df.pop("mask_info_path")
    new_df = pd.merge(new_df,df,left_index=True,right_index=True)
    new_df.to_csv(Path(new_folder)/"trials.csv")

# old_folder = "/media/pierre/NeuroData2/datasets/lot_MEG_encoding/stimulis"
# # # new_folder = "/media/pierre/NeuroData2/datasets/lot_MEG_encoding/stimulis"
# new_folder = "/gpfsscratch/rech/fqt/uzz43va/NeuroData/lot_MEG_encoding/stimulis"
# old_folder ="/media/pierre/NeuroData2/datasets/syntaxicProbingVseamless/UD_ANNfinal"
# new_folder = "/gpfsscratch/rech/fqt/uzz43va/NeuroData/syntaxicProbingVseamless/UD_ANNfinal"
old_folder ="/auto/data5/speechExposureEphys/TCI/TCI_normal/stimulis"
new_folder = "/gpfsscratch/rech/fqt/uzz43va/NeuroData/TCI_normal/stimulis"
for sess in os.listdir(new_folder):
    updateFolders(old_folder+"/"+sess,new_folder+"/"+sess)
# # updateFolders(old_folder,new_folder)

# old_folder = "/media/pierre/NeuroData2/datasets/lot_MEG_encoding/"
# new_folder = "/media/pierre/NeuroData2/datasets/lot_MEG_encoding/stimulis/"
# for sess in os.listdir(new_folder):
#     updateFolders(old_folder+"/"+sess,new_folder+"/"+sess)

# old_folder = "/media/pierre/NeuroData2/datasets/ANN_nsd"
# new_folder = "/media/pierre/NeuroData2/datasets/speechExposure_ltq_nsd/nsd"
# updateFolders(old_folder,new_folder)

# old_folder = "/media/pierre/NeuroData2/datasets/speechExposure_ltq_nsd/nsdreduced"
# new_folder = "/gpfsscratch/rech/fqt/uzz43va/NeuroData/speechExposure_ltq_nsd/sounds/nsdreduced"
# updateFolders(old_folder,new_folder)
#
# old_folder = "/media/pierre/NeuroData2/datasets/speechExposure_ltq_nsd/ltq"
# new_folder = "/gpfsscratch/rech/fqt/uzz43va/NeuroData/speechExposure_ltq_nsd/sounds/ltq"
# updateFolders(old_folder,new_folder)
#
# old_folder = "/media/pierre/NeuroData2/datasets/speechExposure_ltq_nsd/nsd"
# new_folder = "/gpfsscratch/rech/fqt/uzz43va/NeuroData/speechExposure_ltq_nsd/sounds/nsd"
# updateFolders(old_folder,new_folder)
#
# old_folder = "/media/pierre/NeuroData2/datasets/speechExposure_ltq_nsd/ltq"
# new_folder = "/gpfsscratch/rech/fqt/uzz43va/NeuroData/speechExposure_ltq_nsd/sounds/ltq"
# # new_folder = "/media/pierre/NeuroData2/datasets/speechExposure_ltq_nsd/ltq"
# updateFolders(old_folder,new_folder)


# old_folder = "/media/pierre/NeuroData2/datasets/speechExposure_ltq_nsd/nsdsounds"
# new_folder = "/media/pierre/NeuroData2/datasets/speechExposure_ltq_nsd/nsd/sounds"
# # new_folder = "/media/pierre/NeuroData2/datasets/speechExposure_ltq_nsd/ltq"
# updateFolders(old_folder,new_folder)

# # old_folder = "/media/pierre/NeuroData2/datasets/speechExposure_ltq_nsd/ltq"
# old_folder = "/gpfsscratch/rech/fqt/uzz43va/NeuroData/speechExposure_ltq_nsd/sounds/ltq//"
# new_folder = "/gpfsscratch/rech/fqt/uzz43va/NeuroData/speechExposure_ltq_nsd/sounds/ltq/"
# # new_folder = "/media/pierre/NeuroData2/datasets/speechExposure_ltq_nsd/ltq"
# updateFolders(old_folder,new_folder)

#
# old_folder = "/auto/data5/speechExposureEphys/oscipek/ltq/"
# new_folder = "/gpfsscratch/rech/fqt/uzz43va/NeuroData/speechExposure_ltq_nsd/sounds/ltq/"
# updateFolders(old_folder,new_folder)