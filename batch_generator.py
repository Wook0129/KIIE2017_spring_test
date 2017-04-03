import numpy as np


class TrainValBatchGenerator:

    def __init__(self, val_ratio=0.2, *, train_batch_size, val_batch_size, data_handler):
        self._data = data_handler.data
        train_idx, val_idx = self._train_val_idx_split(val_ratio)
        self._train_batch_generator = BatchGenerator(self._data.loc[train_idx],
                                                     train_batch_size, data_handler.var_idx_to_value_idxs)
        self._val_batch_generator = BatchGenerator(self._data.loc[val_idx], 
                                                    val_batch_size, data_handler.var_idx_to_value_idxs)

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
    def __init__(self, data, batch_size, var_idx_to_value_idxs):
        self.data = data
        self.batch_size = batch_size
        self.var_idx_to_value_idxs = var_idx_to_value_idxs
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

            input_value_idxs = self.data.iloc[rand_instance_ids, [x for x in
                                                           idx_exclude_target_idx if x not
                                                           in corrupt_var_idxs]].values

            corrupt_value_idxs = []
            corrupt_vars = self.data.iloc[rand_instance_ids, corrupt_var_idxs].values
            
            for row in corrupt_vars:
                for corrupt_var_idx in row:
                    for values in self.var_idx_to_value_idxs.values():
                        if corrupt_var_idx in values:
                            corrupt_value_idxs.append(values)

            target_value_idxs = []
            target_vars = self.data.iloc[rand_instance_ids, target_var_idx].values

            print(target_vars)
            for target_var_idx in target_vars:
                for values in self.var_idx_to_value_idxs.values():
                    if target_var_idx in values:
                        target_value_idxs.append(values.index(target_var_idx))

            batch.append([input_value_idxs, corrupt_value_idxs, target_value_idxs])

        return batch

