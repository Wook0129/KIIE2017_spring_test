import config
import os
import pandas as pd
from collections import Counter


class DataHandler:

    # Expect Pandas Type Data
    def __init__(self, data):
        self.num_instances = data.shape[0]
        self.num_of_vars = data.shape[1]
        self.var_names = data.columns
        
        # Binning Numerical Variables
        for var_name, var_type in zip(data.dtypes.keys(), data.dtypes.values):
            if var_type != 'object':
                data[var_name] = pd.cut(data[var_name], config.num_bins)

        self.onehot_encoded_data = pd.get_dummies(data).astype(int)
        self.total_num_of_bins = self.onehot_encoded_data.shape[1]
        self.bin_names = self.onehot_encoded_data.columns
        
        self.var_idx_to_bin_idxs = dict()
        self.num_of_bins_by_var = []
        self.proportion_of_bins_by_var = dict()
        for var_idx, var_name in enumerate(self.var_names):
        # Napping Variable index to Related Bin Indexs
            bin_idxs = []
            for bin_idx, bin_name in enumerate(self.bin_names):
                if var_name in bin_name:
                    bin_idxs.append(bin_idx)
            self.var_idx_to_bin_idxs[var_idx] = bin_idxs
            # Count Bin Numbers by Variables
            self.num_of_bins_by_var.append(len(bin_idxs))
        # Calculate Proportions of Bins In Each Variable
            bins_related_with_var = [bin_name for bin_name in self.bin_names if var_name in bin_name]
            self.proportion_of_bins_by_var[var_idx] = list(self.onehot_encoded_data[bins_related_with_var].mean(axis=0))
    
    # Name should be changed to 'save_bin_names'
    def save_metadata(self, LOG_DIR=config.LOG_DIR, filename=config.metadata_filename):
        with open(os.path.join(LOG_DIR, filename), 'w') as f:
            for var_name in self.bin_names:
                f.write(var_name + '\n')
    
    def get_metadata(self):
        metadata = {
            'num_instances' : self.num_instances,
            'num_of_vars' : self.num_of_vars,
            'var_names' : self.var_names,
            'total_num_of_bins' : self.total_num_of_bins,
            'bin_names' : self.bin_names,
            'var_idx_to_bin_idxs' : self.var_idx_to_bin_idxs,
            'num_of_bins_by_var' : self.num_of_bins_by_var,
            'proportion_of_bins_by_var' : self.proportion_of_bins_by_var
        }
        return metadata