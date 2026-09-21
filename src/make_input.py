import pandas as pd
import torch

def make_input(df, variables):
    IDs = list(df['VOL IDs FOR EXPERIMENTAL PURPOSES'].unique())
    ts_data = []
    ts_targets = []
    ts_times = []  
    for idx in IDs:
        ts_this = df.loc[df['VOL IDs FOR EXPERIMENTAL PURPOSES']==idx]
        df_this = torch.tensor(df.loc[df['VOL IDs FOR EXPERIMENTAL PURPOSES']==idx][variables].values, dtype=torch.float32)
        t_points = torch.tensor(ts_this["Age @ sample (yrs, 2 decimal)"].values, dtype=torch.float32)
        target = list(df.loc[df['VOL IDs FOR EXPERIMENTAL PURPOSES']==idx]['CASE CONTROL STATUS'])[0]
        ts_data.append(df_this)
        ts_times.append(t_points)
        ts_targets.append(target)
    ts_targets = torch.tensor(ts_targets, dtype=torch.int64)
    return ts_data, ts_times, ts_targets
