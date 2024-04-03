from torch.utils.data import Dataset
import torch
from typing import List
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as pplot
import tqdm


directory_separator = os.sep

class PoseDataset(Dataset):
    def __init__(self, feature: List[np.ndarray], annot: List[np.ndarray], window_length:int, annot_class:int=None, transform=None, device=torch.device('cpu')) -> None:
        self.feature = feature
        self.annot = annot
        self.window_length =window_length
        self._index = [0]
        self._annot_class = annot_class
        self.valid_seq_index = None
        self._gen_index()
        self.clear_boring_seqs()
        self.transform = transform
        self.device = device

        super().__init__()
    
    def __len__(self):
        return len(self.valid_seq_index)
    
    def __getitem__(self, index:np.ndarray):
        # Locate index
        if isinstance(index,int):
            index = np.array([index])

        real_indices = self.valid_seq_index[index]
        # data chunk indices
        idxs = np.array([np.argmax(self._index>=(real_index+1)) for real_index in real_indices], dtype=int)
        # offset the cumulative length
        start_idxs = real_indices - self._index[idxs-1]
        # collect fdata chunks
        fdatas = [self.feature[idx-1] for idx in idxs]
        # collect fseq
        fseq = np.stack([fdata[start_idx:start_idx+self.window_length,:] for start_idx, fdata in zip(start_idxs, fdatas)], axis=0)
        adatas = [self.annot[idx-1] for idx in idxs]
        aseq = np.stack([np.squeeze(adata[start_idx:start_idx+self.window_length,:]) for start_idx, adata in zip(start_idxs, adatas)], axis=0)
        fseq = torch.tensor(np.squeeze(fseq),dtype=torch.float32,device=self.device)
        aseq = torch.tensor(np.squeeze(aseq),dtype=torch.int64, device=self.device)
        return fseq,aseq
    
    def clear_boring_seqs(self):
        self.valid_seq_index = np.zeros(self._index[-1])
        with tqdm.tqdm(total=self._index[-1]) as pbar:
            index = 0
            while index < self._index[-1]:
                # Locate index
                idx = np.argmax(self._index>=(index+1))
                start_idx = index - self._index[idx-1]
                adata = self.annot[idx-1]
                aseq = np.squeeze(adata[start_idx:start_idx+self.window_length,:])
                # preallocate 
                if not np.all(aseq == aseq[0]):
                    self.valid_seq_index[index] = 1
                    first_different = np.nonzero(aseq!=aseq[0])[0][0] # Find the first entry with different value
                    index += first_different # next
                    pbar.update(first_different)
                else:
                    index+=(self.window_length-1)
                    pbar.update(self.window_length-1)
                
       
        self.valid_seq_index = np.nonzero(self.valid_seq_index)[0]


    
    def _gen_index(self):
        """
        Generate index map: input index locate window from annot
        """
        cumu = 0
        annot_class = 0
        for at,ft in zip(self.annot, self.feature):
            assert at.shape[0] == ft.shape[0], "annotation and features should have the same length!"
            annot_class = np.max((np.max(at), annot_class))
            cumu += np.max((0,1+at.shape[0]-self.window_length))
            self._index.append(cumu)
        self._index = np.array(self._index)
        if self._annot_class is None:
            self._annot_class = annot_class+1
        else:
            assert self._annot_class >= annot_class, "invalid annotation class number. must geq the largest annotation class."
    def get_annot_class(self):
        _, all_annot = self[:]
        _, counts = np.unique(all_annot.cpu(),return_counts=True)
        weights = torch.tensor(1/counts / np.sum(1/counts), device=self.device, dtype=torch.float32)
        annot_class = {"class": self._annot_class, "weight": weights}
        return annot_class
    

def prepare_dataset(
    path: str,
    window_length: int,
    annot_dir: str = "annot.npz",
    feature_dir: str = "feature.npz",
    device = torch.device('cpu')
):
    """
    path: directory containing npz annotation and tracking files
    annot file: dict, each key value is a 2d ndarray, each entry is a annotation (t,1)
    feat file: dict, same key as annot, same shape as annot, each entry is a feature (t, feat_num)
    """
    sclr = StandardScaler()
    annot = np.load(path + directory_separator + "annotation" + directory_separator + annot_dir, allow_pickle=True)
    feature = np.load(path + directory_separator + "tracking" + directory_separator + feature_dir, allow_pickle=True)
    annots, feats = [], []
    for k, i in annot.items():
        fts = feature[k].flatten()
        ats = i.flatten()
        for i, (at, ft) in enumerate(zip(ats, fts)):
            if at.size == 0:
                continue
            if at.shape[0] != ft.shape[0]:
                continue
            annots.append(at)
            if np.mod(i,2) == 0:
                # append partner features
                ft = np.concatenate((ft, fts[i+1]),1)
            else:
                ft = np.concatenate((ft, fts[i-1]),1)
            # normalize
            sclr.fit(ft)
            ft_transformed = sclr.transform(ft)
            feats.append(ft_transformed)
    dataset = PoseDataset(feats, annots, window_length,device=device)
    return dataset

def transition_mat(dataframe, col="annot", transit_by=1, by="id", fill_value=0):
    df_groups = dataframe.groupby(by)
    unique_vals = dataframe[col].unique()
    transit_dataframe = pd.DataFrame(data=0,index=unique_vals,columns=unique_vals)
    count_table = pd.DataFrame(data=0,index=unique_vals,columns=unique_vals)
    for groupname, group in df_groups:
        group["shifted"] = group[col].shift(periods=transit_by,fill_value=fill_value)
        transit_data = pd.crosstab(group[col],group["shifted"])
        transit_dataframe = transit_dataframe.add(transit_data/transit_data.sum(axis=0),fill_value=0)
       
        count_table = count_table.add((transit_data>-1).astype(np.float32),fill_value=0)

    return transit_dataframe.div(count_table)

def count_continous_bouts(dataframe, col="annot",by="id"):
    unique_vals = dataframe[col].unique()
    df_groups = dataframe.groupby(by)
    val_bouth_lengths = {k: np.empty((0,),dtype=int) for k in unique_vals}
    for val in unique_vals:
        for _, group in df_groups:
            seq = group[col].to_numpy(copy=True)
            seq[seq!=val] = -1
            seq[seq==val] = 1
            seq[seq==-1] = 0
            seq_diff = np.diff(np.insert(seq,0,0))
            # 1 start, -1 end
            start_idx = np.nonzero(seq_diff==1)[0]
            end_idx = np.nonzero(seq_diff==-1)[0]
            if seq[-1] == 1:
                end_idx = np.insert(end_idx,-1,len(seq)-1)
            assert len(start_idx) == len(end_idx), "start idx should have the same length with end idx"
            bout_length = end_idx-start_idx
            val_bouth_lengths[val] = np.concatenate((val_bouth_lengths[val],bout_length))
    
    mx_length = max(len(arr) for arr in val_bouth_lengths.values())
    df = pd.DataFrame({k: np.concatenate((val,[np.nan]*(mx_length-len(val)))) for k,val in val_bouth_lengths.items()})
    return df


    
