import numpy as np


class TrainValBatchGenerator:
    # Expect 2-d Integer array
    def __init__(self, data, val_ratio=0.2, *, train_batch_size, val_batch_size, metadata):
        self._data = data
        train_idx, val_idx = self._train_val_idx_split(val_ratio)
        self._train_batch_generator = BatchGenerator(data[train_idx], train_batch_size,
                                                     metadata)
        self._val_batch_generator = BatchGenerator(data[val_idx], val_batch_size, metadata)

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
    def __init__(self, data, batch_size, metadata):
        self.data = data
        self.batch_size = batch_size
        self.num_of_vars = metadata['num_of_vars']
        self.num_of_bins_by_vars = metadata['num_of_bins_by_var']
        self.var_idx_to_bin_idxs = metadata['var_idx_to_bin_idxs']
        self.proportion_of_bins_by_var = metadata['proportion_of_bins_by_var']
        self.cum_num_of_bins = 0
        self.iter = self.make_random_iter()

    def make_random_iter(self):
        splits = np.arange(self.batch_size, len(self.data), self.batch_size)
        it = np.split(np.random.permutation(range(len(self.data))), splits)[:-1]
        return iter(it)

    def next_batch(self):
        try:
            rand_instance_ids = next(self.iter)
        except StopIteration:
            self.iter = self.make_random_iter()
            rand_instance_ids = next(self.iter)

        self.cum_num_of_bins = 0
        list_of_samples_per_var = []

        for var_idx, num in enumerate(self.num_of_bins_by_vars):
            idx_range = [i for i in range(0, sum(self.num_of_bins_by_vars))]
            input_vars_idx_range = idx_range[: self.cum_num_of_bins] + idx_range[
                                                                       self.cum_num_of_bins + num:]
            target_vars_idx_range = self.var_idx_to_bin_idxs[var_idx]

            input_instances = []
            target_instances = []
            for instance_id in rand_instance_ids:
                instance = self.data[instance_id]
                input_instances.append(
                    [i for i, x in enumerate(instance) if (i in input_vars_idx_range)
                     and (x==1)])
            for instance_id in rand_instance_ids:
                instance = self.data[instance_id]
                target_instances.append(
                    [x for i, x in enumerate(instance) if i in target_vars_idx_range])

            inputs = []
            targets = []
            for input_instance in input_instances:

                # Randomly Corrupt a Variable
                var_idxs = [i for i in range(0, self.num_of_vars)]
                corrupt_var_idx = np.random.choice(
                    var_idxs[:var_idx] + var_idxs[var_idx + 1:])
                corrupt_bin_idxs = self.var_idx_to_bin_idxs[corrupt_var_idx]

                input_var_idxs = []
                for idx in input_instance:
                    if idx not in corrupt_bin_idxs:
                        input_var_idxs.append(idx)
                input_var_idxs.append(self.var_idx_to_bin_idxs[corrupt_var_idx])
                input_var_idxs.append(self.proportion_of_bins_by_var[corrupt_var_idx])
                inputs.append(input_var_idxs)

            for target_instance in target_instances:
                for idx, value in enumerate(target_instance):
                    if value == 1:
                        targets.append(idx)
                        break

            list_of_samples_per_var.append([inputs, targets])
            self.cum_num_of_bins += num

        return list_of_samples_per_var
