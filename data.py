from functools import partial

import numpy as np

from gre.consts import POS_DI, NEG_DI


class Dataset(object):
    def __init__(self, train_path, valid_path, test_path, graph, add_reverse=False):
        train = self._load_examples(train_path)
        valid = self._load_examples(valid_path)
        test = self._load_examples(test_path)

        if add_reverse:
            train = self._add_reverse_examples(train)
            valid = self._add_reverse_examples(valid)
            test = self._add_reverse_examples(test)

        self.train = self._convert_to_numpy(train, graph.v_str2id, graph.et_str2id)  # (2D numpy) n_train x 4
        self.valid = self._convert_to_numpy(valid, graph.v_str2id, graph.et_str2id)  # (2D numpy) n_valid x 4
        self.test = self._convert_to_numpy(test, graph.v_str2id, graph.et_str2id)  # (2D numpy) n_test x 4

    @staticmethod
    def _load_examples(path):
        examples = []
        with open(path) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                examples.append((h, t, r, POS_DI))
            return examples

    @staticmethod
    def _add_reverse_examples(examples):
        return examples + [(t, h, r, NEG_DI) for h, t, r, _ in examples]

    @staticmethod
    def _convert_to_numpy(examples, v_str2id, et_str2id):
        return np.array([[v_str2id[h], v_str2id[t], et_str2id[r], di] for h, t, r, di in examples])

    @staticmethod
    def _train_loader(train_data, batch_size):
        n_train = len(train_data)
        rand_idx = np.random.permutation(n_train)
        start = 0
        while start < n_train:
            end = min(start + batch_size, n_train)
            pad = max(start + batch_size - n_train, 0)
            batch = train_data[np.concatenate([rand_idx[start:end], rand_idx[:pad]])]
            yield batch, end - start
            start = end

    @staticmethod
    def _eval_loader(eval_data, batch_size):
        n_eval = len(eval_data)
        start = 0
        while start < n_eval:
            end = min(start + batch_size, n_eval)
            batch = eval_data[start:end]
            yield batch, end - start
            start = end

    def get_train_loader(self):
        return partial(self._train_loader, self.train)

    def get_valid_loader(self):
        return partial(self._eval_loader, self.valid)

    def get_test_loader(self):
        return partial(self._eval_loader, self.test)

    @property
    def n_train(self):
        return len(self.train)

    @property
    def n_valid(self):
        return len(self.valid)

    @property
    def n_test(self):
        return len(self.test)


class FB237(Dataset):
    def __init__(self, graph):
        self.name = 'FB237'
        train_path = 'datasets/FB237/train'
        valid_path = 'datasets/FB237/valid'
        test_path = 'datasets/FB237/test'
        super(FB237, self).__init__(train_path, valid_path, test_path, graph, add_reverse=True)
