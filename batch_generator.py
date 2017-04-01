import numpy as np


class TrainValBatchGenerator:

    # Expect 2-d Integer array
    def __init__(self, data, val_ratio=0.2, *, train_batch_size, val_batch_size, num_of_values_in_vars):
        self._data = data
        train_idx, val_idx = self._train_val_idx_split(val_ratio)
        self._train_batch_generator = BatchGenerator(data[train_idx], train_batch_size, num_of_values_in_vars)
        self._val_batch_generator = BatchGenerator(data[val_idx], val_batch_size, num_of_values_in_vars)
        self._num_of_values_in_vars = num_of_values_in_vars
    
    def _train_val_idx_split(self, val_ratio):
        idxs = np.arange(len(self._data))
        np.random.shuffle(idxs)
        split_idx = int((1-val_ratio) * len(idxs))
        train_idx = idxs[:split_idx]
        val_idx = idxs[split_idx:]
        return train_idx, val_idx
    
    def next_train_batch(self):
        return self._train_batch_generator.next_batch()
    
    def next_val_batch(self):
        return self._val_batch_generator.next_batch()


class BatchGenerator:

    def __init__(self, data, batch_size, num_of_values_in_vars):
        self.data = data
        self.batch_size = batch_size
        self.num_of_values_in_vars = num_of_values_in_vars
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

        list_of_samples_per_var = []

        for var_idx, num in enumerate(self.num_of_values_in_vars):
            idx_range = [i for i in range(0, sum(self.num_of_values_in_vars))]
            input_vars_idx_range = idx_range[ : self.cum_num_of_bins] + idx_range[self.cum_num_of_bins + num : ]
            target_vars_idx_range = idx_range[self.cum_num_of_bins : self.cum_num_of_bins + num]
            
            input_instances = []
            target_instances = []
            for instance_id in rand_instance_ids:
                instance = self.data[instance_id]
                input_instances.append([x for i, x in enumerate(instance) if i in input_vars_idx_range])
            for instance_id in rand_instance_ids:
                instance = self.data[instance_id]
                target_instances.append([x for i, x in enumerate(instance) if i in target_vars_idx_range])
            
            inputs = []
            targets = []
            
            for input_instance, instance_id in zip(input_instances, rand_instance_ids):
                input_var_idxs = []
                for idx, value in enumerate(input_instance):
                    if value == 1:
                        input_var_idxs.append(idx)
                input_var_idxs.append(instance_id)
                inputs.append(input_var_idxs)
                
            for target_instance in target_instances:
                for idx, value in enumerate(target_instance):
                    if value == 1:
                        targets.append([idx])
                        break
                        
            list_of_samples_per_var.append([np.array(inputs), np.array(targets)])
            self.cum_num_of_bins += num
            
        return list_of_samples_per_var
