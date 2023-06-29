import copy
import itertools
import lightning.pytorch as pl
import torch

from .datasets import FrDataset, FrTxtDataset, GradPerpDemoDataset
from src.utils import load_split_df, load_tx_df, load_text_df
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class BaseDataModule(pl.LightningDataModule):
    '''Base DataModule

    Implement `train_dataloader`, `val_dataloader` and `test_dataloader`

    The `collate_fn` and `setup` methods need to be overridden:
        setup: initialize self.train_dataset, self.val_dataset and self.test_dataset
    '''

    def __init__(self,
                 tasks,
                 num_workers,
                 batch_size,
                 val_batch_size,
                 test_batch_size,
                 pin_memory):

        super().__init__()

        self.tasks = tasks
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        pass

    def collate_fn(self, data):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=False,
            persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=False,
            persistent_workers=True)


class FrDataModule(BaseDataModule):
    '''DataModule with ONLY financial ratios

    Override the `setup` and `collate_fn` methods

    Args:
        train_val_split: List, [train_size, val_size]
    '''

    def __init__(self, tx_df_name, split_df_name, tasks, num_workers, bsz, val_bsz, test_bsz,
                 pin_memory, split_id, train_val_split, use_test_as_val, data_dir, **kwargs):

        super().__init__(tasks=tasks,
                         num_workers=num_workers,
                         batch_size=bsz,
                         val_batch_size=val_bsz,
                         test_batch_size=test_bsz,
                         pin_memory=pin_memory)

        self.tx_df_name = tx_df_name
        self.split_df_name = split_df_name
        self.tasks = tasks
        self.num_workers = num_workers
        self.batch_size = bsz
        self.val_batch_size = val_bsz
        self.test_batch_size = test_bsz
        self.data_dir = data_dir
        self.train_val_split = train_val_split
        self.use_test_as_val = use_test_as_val
        self.split_id = split_id

    # create dataset
    def setup(self, stage=None):

        # Create train/val Dataset
        self.train_val_dataset = FrDataset(tx_df_name=self.tx_df_name,
                                           split_df_name=self.split_df_name,
                                           tasks=self.tasks,
                                           split_type='train_val',
                                           split_id=self.split_id,
                                           data_dir=self.data_dir)

        n_train_val_split = [round(x*len(self.train_val_dataset))
                             for x in self.train_val_split]

        self.train_dataset, self.val_dataset = random_split(
            self.train_val_dataset, n_train_val_split,
            generator=torch.Generator().manual_seed(42))

        # create test Dataset
        self.test_dataset = FrDataset(tx_df_name=self.tx_df_name,
                                      split_df_name=self.split_df_name,
                                      split_type='test',
                                      split_id=self.split_id,
                                      tasks=self.tasks,
                                      data_dir=self.data_dir)

        # override val with test if realy needed
        if self.use_test_as_val:
            self.val_dataset = copy.deepcopy(self.test_dataset)

        # get N obs of each Dataset
        self.n_train_dataset = len(self.train_dataset)
        self.n_val_dataset = len(self.val_dataset)
        self.n_test_dataset = len(self.test_dataset)

        print(f"\n{'-'*5} Dataset with split \"{self.split_id}\" {'-'*5}")
        print(f'N train = {len(self.train_dataset)}')
        print(f'N val = {len(self.val_dataset)}')
        print(f'N test = {len(self.test_dataset)}')
        print(f"{'-'*46}\n")

    # Collate_fn
    def collate_fn(self, data):
        # Unpack a batch
        docid_idx, t, mantxts, finratios, \
            auxcars, auxvols,\
            fund, revision, retail = zip(*data)

        return {
            'docid_idx': torch.tensor(docid_idx, dtype=torch.int64),
            't': torch.tensor(t, dtype=torch.float16),
            'mantxts': torch.tensor(mantxts, dtype=torch.float16),
            'finratios': torch.tensor(finratios, dtype=torch.float16),
            'auxcars': torch.tensor(auxcars, dtype=torch.float16),
            'auxvols': torch.tensor(auxvols, dtype=torch.float16),
            'fund': torch.tensor(fund, dtype=torch.float16),
            'revision': torch.tensor(revision, dtype=torch.float16),
            'retail': torch.tensor(retail, dtype=torch.float16),
        }


class FrTxtDataModule(BaseDataModule):
    '''DataModule with ONLY financial ratios

    Override the `setup` and `collate_fn` methods

    Compared with FrDataModule, the following new states are added:
        tokenizer: a transformer tokenizer
    '''

    def __init__(self, tx_df_name, split_df_name, preemb_dir,
                 split_id,
                 model_cfg, tasks, num_workers,
                 bsz, val_bsz, test_bsz, pin_memory, train_val_split,
                 use_test_as_val, data_dir, dataset_txt_return_type,
                 datamodule_txt_return_type, max_doc_len, **kwargs):
        '''
        Args:
            type_tokens: List. Choices of type tokens
                1: MD
                2: Q
                3: A
        '''

        super().__init__(tasks=tasks,
                         num_workers=num_workers,
                         batch_size=bsz,
                         val_batch_size=val_bsz,
                         test_batch_size=test_bsz,
                         pin_memory=pin_memory)

        # --------------------
        # configs sanity check
        # --------------------
        assert dataset_txt_return_type in ['preemb']
        assert datamodule_txt_return_type in [
            'padded_tensor', 'packed_tensor']

        if dataset_txt_return_type == 'preemb':
            assert datamodule_txt_return_type in [
                'padded_tensor', 'packed_tensor']
            assert preemb_dir is not None

        # states that are UNIQUE to FrTxtDataModule
        self.d_model = model_cfg.d_model
        self.preemb_dir = preemb_dir
        self.model_cfg = model_cfg
        self.dataset_txt_return_type = dataset_txt_return_type
        self.datamodule_txt_return_type = datamodule_txt_return_type
        self.max_doc_len = max_doc_len

        # states that are also in FrDataModule
        self.tx_df_name = tx_df_name
        self.split_df_name = split_df_name
        self.tasks = tasks
        self.num_workers = num_workers
        self.batch_size = bsz
        self.val_batch_size = val_bsz
        self.test_batch_size = test_bsz
        self.data_dir = data_dir
        self.train_val_split = train_val_split
        self.use_test_as_val = use_test_as_val
        self.split_id = split_id

    # Initialize Dataset
    def setup(self, stage=None):

        # Create train/val Dataset
        self.train_val_dataset = FrTxtDataset(
            split_type='train_val',
            split_id=self.split_id,
            tx_df_name=self.tx_df_name,
            split_df_name=self.split_df_name,
            d_model=self.d_model,
            preemb_dir=self.preemb_dir,
            tasks=self.tasks,
            dataset_txt_return_type=self.dataset_txt_return_type,
            data_dir=self.data_dir)

        n_train_val_split = [round(x*len(self.train_val_dataset))
                             for x in self.train_val_split]

        self.train_dataset, self.val_dataset = random_split(
            self.train_val_dataset, n_train_val_split,
            generator=torch.Generator().manual_seed(42))

        # create test Dataset
        self.test_dataset = FrTxtDataset(
            split_type='test',
            split_id=self.split_id,
            tx_df_name=self.tx_df_name,
            split_df_name=self.split_df_name,
            d_model=self.d_model,
            preemb_dir=self.preemb_dir,
            tasks=self.tasks,
            dataset_txt_return_type=self.dataset_txt_return_type,
            data_dir=self.data_dir)

        # override val with test if realy needed
        if self.use_test_as_val:
            self.val_dataset = copy.deepcopy(self.test_dataset)

        # get N obs of each Dataset
        self.n_train_dataset = len(self.train_dataset)
        self.n_val_dataset = len(self.val_dataset)
        self.n_test_dataset = len(self.test_dataset)

        print(f"\n{'-'*5} Dataset with split \"{self.split_id}\" {'-'*5}")
        print(f'N train = {len(self.train_dataset)}')
        print(f'N val = {len(self.val_dataset)}')
        print(f'N test = {len(self.test_dataset)}')
        print(f"{'-'*46}\n")

    # Collate_fn
    def collate_fn(self, data):
        '''
        We'll do tokenization (if needed) here
        '''
        # unpack a batch
        docid_idx, t, mantxts, finratios, \
            auxcars, auxvols,\
            fund, revision, retail,\
            docs = zip(*data)

        # return pre-embeddings

        doc_preembs = {}
        doc_typetokens = {}

        # iter over each type_token
        section_type_token_map = {'md': 1, 'qa': 2}

        for section_type in ['md', 'qa']:
            section_token = section_type_token_map[section_type]

            # trim to max_doc_len
            doc_lens = [min(d[section_type].size(0), self.max_doc_len)
                        for d in docs]

            max_doc_len = max(doc_lens)

            # trim embs to maximum length
            doc_embs_trimmed = [d[section_type][:max_doc_len, :] for d in docs]

            # creawte type tokens, which are the same shape as the embs
            doc_typetokens_trimmed = [torch.tensor(
                [section_token]*d.size(0), dtype=torch.int64) for d in doc_embs_trimmed]

            # return padded tensor (for TSFM)
            if self.datamodule_txt_return_type == 'padded_tensor':
                # pad
                doc_embs_trimmed_padded = torch.nn.utils.rnn.pad_sequence(
                    doc_embs_trimmed, batch_first=True)  # (N, S, E)

                doc_typetokens_trimmed_padded = torch.nn.utils.rnn.pad_sequence(
                    doc_typetokens_trimmed, batch_first=True)

                # maks: True (masked), False (not masked)
                attention_mask = torch.tensor([
                    [False]*l + [True]*(max_doc_len-l)
                    for l in doc_lens
                ], dtype=torch.bool)

                # add to output
                doc_preembs[section_type] = {
                    'input_embeddings': doc_embs_trimmed_padded,
                    'attention_mask': attention_mask,
                }
                doc_typetokens[section_type] = doc_typetokens_trimmed_padded

            # return packed tensor (for GRU)
            elif self.datamodule_txt_return_type == 'packed_tensor':
                # pack
                doc_embs_trimmed_packed = torch.nn.utils.rnn.pack_sequence(
                    doc_embs_trimmed, enforce_sorted=False)

                # add to output
                doc_preembs[section_type] = {
                    'input_embeddings': doc_embs_trimmed_packed,
                }

        return {
            'docid_idx': torch.tensor(docid_idx, dtype=torch.int64),
            't': torch.tensor(t, dtype=torch.float16),
            'mantxts': torch.tensor(mantxts, dtype=torch.float16),
            'finratios': torch.tensor(finratios, dtype=torch.float16),
            'auxcars': torch.tensor(auxcars, dtype=torch.float16),
            'auxvols': torch.tensor(auxvols, dtype=torch.float16),
            'fund': torch.tensor(fund, dtype=torch.float16),
            'revision': torch.tensor(revision, dtype=torch.float16),
            'retail': torch.tensor(retail, dtype=torch.float16),
            'doc_preembs': doc_preembs,
            'doc_typetokens': doc_typetokens
        }

    def _trim_to_max_doc_len(self, doc_tokens=None, doc_embs=None):
        '''Trim inputs to max_doc_len

        Make sure that the N of sentences in a document is no greater than max_doc_len
        '''
        assert (doc_tokens is None) + (doc_embs is None) == 1

        overflow_to_sample_mapping = doc_tokens['overflow_to_sample_mapping']
        input_ids = doc_tokens['input_ids']
        attention_mask = doc_tokens['attention_mask']

        # get selector
        _, counts = torch.unique_consecutive(
            overflow_to_sample_mapping, return_counts=True)

        doc_lens = [min(int(n), self.max_doc_len) for n in counts]  # (N,)

        selector = [[True]*min(self.max_doc_len, n) + [False]
                    * max(0, n-self.max_doc_len) for n in counts]
        selector = list(itertools.chain(*selector))

        # trim according to selector
        input_ids = input_ids[selector, :]
        attention_mask = attention_mask[selector, :]

        return input_ids, attention_mask, doc_lens


class GradPerpDemoDataModule(pl.LightningDataModule):
    def __init__(self, bsz, tasks, **kwargs):
        super().__init__(self)

        self.bsz = bsz
        self.tasks = tasks

    def setup(self, stage=None):
        dataset = GradPerpDemoDataset(tasks=self.tasks)
        train_size = int(len(dataset)*0.8)
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.bsz,
            num_workers=4,
            persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.bsz,
            num_workers=4,
            persistent_workers=True)
