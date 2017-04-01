import config
import os
import pandas as pd
from collections import Counter


class DataHandler:
    # Expect Pandas Type Data
    def __init__(self, data):
        self.num_instances = data.shape[0]
        self.num_of_vars = data.shape[1]

        for var_name, var_type in zip(data.dtypes.keys(), data.dtypes.values):
            if var_type != 'object':
                data[var_name] = pd.cut(data[var_name], config.num_bins)

        self.onehot_encoded_data = pd.get_dummies(data).astype(int)
        self.total_num_of_bins = self.onehot_encoded_data.shape[1]
        self.var_names = self.onehot_encoded_data.columns

        dup_var_names = [var_name.split('_')[0] for var_name in self.var_names]
        var_name_count = Counter(dup_var_names)

        for i, x in enumerate(dup_var_names):
            while x in dup_var_names[i + 1:]:
                dup_var_names.remove(x)

        self.num_of_bins_by_var = [var_name_count[var_name] for var_name in dup_var_names]

    def save_metadata(self, LOG_DIR=config.LOG_DIR, filename=config.metadata_filename):
        with open(os.path.join(LOG_DIR, filename), 'w') as f:
            for var_name in self.var_names:
                f.write(var_name + '\n')