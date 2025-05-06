import os
import numpy as np
from pathlib import Path
from sounds.perExperiment.protocols.AlRoumi2023 import LOT_deviants
from sounds.perExperiment.protocols.ProtocolGeneration import ListProtocol_independentTrial
from sounds.experimentsClass.element_masking import mask_and_latent_BalancedNegatives
import mne
import scipy.signal
import pandas as pd
import regex


### We construct an encoding dataset for the LOT_MEG experiments.
## To simplify, we can gather the order with which the patient was presented the sequences
# and recreate for each trial, the order of deviants, with the good position of deviants.

sequence_isi = 0.1 #0.5
isi = 0 #0.2

# main_dir = Path("/media/pierre/NeuroData2/datasets/LOT_MEG/BIDS_deriv")
main_dir = Path("/auto/data5/fusExposure/LOT_MEG/BIDS_deriv/")
for subject in list(filter(lambda e:e.startswith("sub-"),os.listdir(main_dir))):
    try:
        ds = mne.read_epochs(main_dir /subject /"meg" /(subject +"_task-abseq_proc-icaDetrend_epo.fif"),
                                preload=False)
    except:
        print(subject)
        raise  Exception("")
    ## RegEx to read the values:
    myregex = r"SequenceID_(?P<seqid>\d+)/" + "RunNumber_(?P<run>\d+)/" + "TrialNumber_(?P<trialnum>\d+)/" + "StimPosition_(?P<stimpos>\d+)/" \
              + "StimID_(?P<stimid>\d+)/" + "ViolationOrNot_(?P<hasdeviant>\d+)/" + "Complexity_(?P<complexity>\d+)/" + "ViolationInSequence_(?P<seqhasdeviant>\d+)"
    all_run = [regex.match(myregex, o) for o in list(ds.event_id.keys())]
    all_values = {k: np.array([int(a.group(k)) for a in all_run]) for k in
                  ["seqid", "run", "trialnum", "stimpos", "stimid", "hasdeviant", "complexity", "seqhasdeviant"]}
    all_values["event_id"] = np.array(list(ds.event_id.values()))
    df = pd.DataFrame.from_dict(all_values)

    f0 = 350
    f1 = 500
    seqs = ["repeat","alternate","pairs","quadruplets","pairsAndAlt1","shrinking","complex"]
    rs = []
    for seqid in range(1,len(seqs)+1):
        filter_id = (df["seqid"] == seqid)
        correct_id = df[filter_id]["event_id"]
        correct_events = np.where(np.any(np.stack([ds.events[:, -1] == v for v in correct_id], axis=0), axis=0))[0]
        id_to_events = [np.where(ds.events[:, -1] == v)[0] for v in correct_id]
        sort_events = np.argsort(
            df[filter_id]["run"].values * 100000 + df[filter_id]["trialnum"].values * 1000 + df[filter_id]["stimpos"].values)
        try:
            is_deviant = df[filter_id]["seqhasdeviant"].values[sort_events].reshape(2,-1,16)
            assert np.all(np.all(is_deviant == 0, axis=-1) + np.all(is_deviant > 1,
                                                                    axis=-1))  # verify same deviance in one sequence
            deviant_seq = np.any(is_deviant > 0, axis=-1)
            deviant_pos = is_deviant[..., 0]
            rs += [LOT_deviants(sequence_isi=sequence_isi, isi=isi,
                                name="LOT",
                                tones_fs=np.array([[f0, f0 * 2, f0 * 4], [f1, f1 * 2, f1 * 4]]),
                                duration_tone=0.05, samplerate=16000,
                                lot_seq=seqs[seqid - 1],
                                motif_repeat=deviant_seq[0, :].shape[0],
                                pos_deviants_sequences=np.where(deviant_seq[0, :])[0],
                                pos_deviants_in_pattern=deviant_pos[0, np.where(deviant_seq[0, :])[0]])]
            rs += [LOT_deviants(sequence_isi=sequence_isi, isi=isi,
                                name="LOTcomplementary",
                                tones_fs=np.array([[f1, f1 * 2, f1 * 4], [f0, f0 * 2, f0 * 4]]),
                                duration_tone=0.05, samplerate=16000,
                                lot_seq=seqs[seqid - 1],
                                motif_repeat=deviant_seq[0, :].shape[0],
                                pos_deviants_sequences=np.where(deviant_seq[0, :])[0],
                                pos_deviants_in_pattern=deviant_pos[0, np.where(deviant_seq[0, :])[0]])]
        except:
            print("problem in seq_id ",seqid," for subject ",subject)
    if len(rs)>0:
        output_dir = Path("/media/pierre/NeuroData2/datasets/lot_MEG_encoding2/stimulis") / subject
        os.makedirs(output_dir,exist_ok=True)
        lp = ListProtocol_independentTrial(rs)
        lp.generate(n_trial=1,output_dir=output_dir)
        mask_and_latent_BalancedNegatives(str(output_dir))