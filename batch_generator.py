import numpy as np


class TrainValBatchGenerator:

    # Expect 2-d Integer array
    def __init__(self, data, val_ratio=0.2, *, train_batch_size, val_batch_size):
        self._data = data
        train_idx, val_idx = self._train_val_idx_split(val_ratio)
        self._train_batch_generator = BatchGenerator(data[train_idx], train_batch_size)
        self._val_batch_generator = BatchGenerator(data[val_idx], val_batch_size)
    
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

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
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
            
        # Make n Samples with One Instance
        def make_samples_with(instance):
            inputs = []
            target = []            
            variable_idxs = []
            for idx, value in enumerate(instance):
                if value == 1:
                    variable_idxs.append(idx)
            for i, idx in enumerate(variable_idxs):
                inputs.append(variable_idxs[:i] + variable_idxs[i+1:])
                target.append([idx])
            return inputs, target
        
        X = []
        Y = []
        for instance_id in rand_instance_ids:
            inputs, target = make_samples_with(self.data[instance_id])
            for i, _ in enumerate(inputs):
                inputs[i] += [instance_id] # Add Instance ID to input
            X += inputs
            Y += target
            
        return np.array(X), np.array(Y)
