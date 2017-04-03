import numpy as np


class TrainValBatchGenerator:

    def __init__(self, val_ratio=0.2, *, train_batch_size, val_batch_size, data_handler):
        self._data = data_handler.data
        train_idx, val_idx = self._train_val_idx_split(val_ratio)
        self._train_batch_generator = BatchGenerator(self._data.loc[train_idx],
                                                     train_batch_size)
        self._val_batch_generator = BatchGenerator(self._data.loc[val_idx], val_batch_size)

    def _train_val_idx_split(self, val_ratio):
        idxs = np.arange(len(self._data))
        np.random.shuffle(idxs)
        split_idx = int((1 - val_ratio) * len(idxs))
        train_idx = idxs[:split_idx]
        val_idx = idxs[split_idx:]
        return train_idx, val_idx

    def next_train_batch(self):
        return self._train_batch_generator.next_batch()

    def next_val_batch(self):
        return self._val_batch_generator.next_batch()


class BatchGenerator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.iter = self.make_random_iter()

    def make_random_iter(self):
        splits = np.arange(self.batch_size, len(self.data), self.batch_size)
        it = np.split(np.random.permutation(range(len(self.data))), splits)[:-1]
        return iter(it)

    def next_batch(self, corruption_ratio=0.1):
        try:
            rand_instance_ids = next(self.iter)
        except StopIteration:
            self.iter = self.make_random_iter()
            rand_instance_ids = next(self.iter)

        batch = []

        num_of_vars = self.data.shape[1]
        num_corruption = int(num_of_vars * corruption_ratio)

        for target_var_idx in range(num_of_vars):
            # Randomly Corrupt Variables

            var_idxs = [i for i in range(0, num_of_vars)]
            idx_exclude_target_idx = var_idxs[:target_var_idx] + var_idxs[target_var_idx + 1:]
            corrupt_var_idxs = np.random.choice(idx_exclude_target_idx, num_corruption,
                                                replace=False)

            input_vars = self.data.iloc[rand_instance_ids, [x for x in
                                                           idx_exclude_target_idx if x not
                                                           in corrupt_var_idxs]]
            corrupt_vars = self.data.iloc[rand_instance_ids, corrupt_var_idxs]
            target_var = self.data.iloc[rand_instance_ids, target_var_idx]

            batch.append([input_vars.values, corrupt_vars.values, target_var.values])

        return batch


#TODO corrupt랑 target에서 variable_value_index가 뽑히면 같은 varaible_index에 해당하는,
#variable_value_index로 가지고 오기

#TODO target을 0~len(variable_value_index)로 바꾸기

import pandas as pd
from data_handler import DataHandler
data = pd.read_csv('data/Mushroom.csv').drop('class',axis=1)
data_handler = DataHandler(data)
gen = TrainValBatchGenerator(train_batch_size=3, val_batch_size=1,
                             data_handler=data_handler)
temp = gen.next_train_batch()

for num, i in enumerate(temp):
    input_var, corrupt_var, target_var = i
    if sum([len(x) for x in input_var]) != 57:
        print(num, 'he')
    assert sum([len(x) for x in input_var]) == 57
    assert sum([len(x) for x in corrupt_var]) == 6
    assert len(target_var) == 3

