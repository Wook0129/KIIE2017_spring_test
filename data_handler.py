from collections import Counter


class DataHandler:
    # Expect Pandas Type Data
    def __init__(self, data):
        self.data, self.variable_value_dictionary = self.preprocess_for_embedding(data)
        self.num_instances = data.shape[0]
        self.num_of_vars = data.shape[1]
        self.var_names = data.columns
        self.num_of_values_by_var = [len(x) for x in
                                    self.variable_value_dictionary.values()]
        self.total_num_of_bins = sum(self.num_of_values_by_var)

        self.proportion_of_bins_by_var = dict()
        for i, values in self.variable_value_dictionary.items():
            value_count = list(Counter(self.data.iloc[:,i]).values())
            total_count = sum(value_count)
            value_proportions = [count / total_count for count in value_count]
            self.proportion_of_bins_by_var[i] = value_proportions

    def preprocess_for_embedding(self, data):
        unique_dict = dict()
        index_counter = 0
        for i in range(data.shape[1]):
            tmp = data.iloc[:, i]
            unique_cat = set(tmp)
            for val in unique_cat:
                tmp.replace(val, index_counter, inplace=True)
                index_counter += 1
            unique_dict[i] = unique_cat
        return data, unique_dict
