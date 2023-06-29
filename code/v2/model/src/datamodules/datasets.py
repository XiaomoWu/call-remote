import numpy as np
import logging
import torch

from torch.utils.data import Dataset
from pyarrow.feather import read_feather
from src.utils import load_split_df, load_tx_df

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    '''Base Dataset

    BaseDataset implement some helpers:

        _get_manual_texts: return text manual features (sentiment, similarity)
        _get_financial_ratios: return the financial ratios

    BaseDataset also initialize some states:
        split_type
        fr_df (all targets and featuers EXCLUDING texts)
        [train/val/test]_date

    The `__getitem__` method needs to be overridden
    '''

    def __init__(self, tasks, tx_df_name, split_df_name, split_type, split_id, data_dir):
        '''
        Args:
            tx_df: DataFrame of CARs, financial ratios and manual text features. 
                Does not include text

            tasks: list of str. The name of target variables

            split_id: a string that indicates the start/end date of the the train/val/test datasets.
                e.g., '08/20-21' (train: 2008-2020, test: 2021)

            split_type: "train_val" or "test." A string indicating if the output is train_val or test

            split_df: DataFrame that keeps the split of windows

            split_type: str. 'train', 'val', or 'test'

        '''

        # load target and features
        tx_df = load_tx_df(tx_df_name, tasks, data_dir)

        # load split_df
        split_df = load_split_df(split_id, split_df_name, data_dir)

        # -------------------------------
        # get split dates
        # -------------------------------

        # get start/end dates of train/val/test
        _, train_start_date, train_end_date, test_start_date, test_end_date, _ = \
            tuple(split_df.loc[(split_df.split_id == split_id)].iloc[0])

        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.split_type = split_type

        # -------------------------------
        # get Targets and Non-text features
        # for train/val/test
        # -------------------------------

        if split_type == 'train_val':
            tx_df = tx_df[tx_df.rdq.between(
                train_start_date, train_end_date)]

        elif split_type == 'test':
            tx_df = tx_df[tx_df.rdq.between(
                test_start_date, test_end_date)]

        # -------------------------------
        # assign other states
        # -------------------------------
        self.tx_df_name = tx_df_name
        self.tx_df = tx_df
        self.tasks = tasks

    def __len__(self):
        return len(self.tx_df)

    def __getitem__(self, idx):
        pass

    def _get_auxcars(self, sample):
        '''
        This function extract auxiliary cars (CAR5, CAR10, CAR20) from the given data sample point
        '''
        if 'tx_v1' in self.tx_df_name:
            car_0_5_stand = sample.car_0_5_stand
            car_0_10_stand = sample.car_0_10_stand
            car_0_20_stand = sample.car_0_20_stand
            return (car_0_5_stand, car_0_10_stand, car_0_20_stand)
        elif 'tx_v2' in self.tx_df_name:
            car_dgtw_call_0_3_std = sample.car_dgtw_call_0_3_std
            car_dgtw_call_0_21_std = sample.car_dgtw_call_0_21_std
            car_dgtw_call_0_63_std = sample.car_dgtw_call_0_63_std
            return (car_dgtw_call_0_3_std, car_dgtw_call_0_21_std, car_dgtw_call_0_63_std)

    def _get_auxvols(self, sample):
        '''
        This function extract auxiliary vols (vol5, vol10, vol20) from the given data sample point
        '''
        if 'tx_v1' in self.tx_df_name:
            vol_0_5_stand = sample.vol_0_5_stand
            vol_0_10_stand = sample.vol_0_10_stand
            vol_0_20_stand = sample.vol_0_20_stand
            return (vol_0_5_stand, vol_0_10_stand, vol_0_20_stand)
        elif 'tx_v2' in self.tx_df_name:
            vol = sample.vol_call_0_63_std
            return (vol,)

    def _get_fund(self, sample):
        '''
        This function extract inflow from the given data sample point
        '''
        if 'tx_v1' in self.tx_df_name:
            inflow_0_90_stand = sample.inflow_0_90_stand
            return (inflow_0_90_stand,)
        elif 'tx_v2' in self.tx_df_name:
            fund_0_90_std = sample.fund_0_90_std
            return (fund_0_90_std,)

    def _get_revision(self, sample):
        '''
        This function extract revision from the given data sample point
        '''
        if 'tx_v1' in self.tx_df_name:
            revision = sample.revision_0_90_stand
        elif 'tx_v2' in self.tx_df_name:
            revision = sample.revision_scaled_by_price_90_std
        return (revision,)

    def _get_retail(self, sample):
        '''
        This function extract retail from the given data sample point
        '''
        if 'tx_v1' in self.tx_df_name:
            retail = sample.retail_net_0_3_stand
        elif 'tx_v2' in self.tx_df_name:
            retail = sample.demand_retail_3_std
        return (retail,)

    def _get_manual_texts(self, sample):
        '''
        This function extract manual text featurs from the given
        data sample point
        '''
        if 'tx_v1' in self.tx_df_name:
            similarity = sample.similarity_bigram_stand
            sentiment_negative_qa_analyst = sample.sentiment_negative_qa_analyst
            return (similarity, sentiment_negative_qa_analyst)
        elif 'tx_v2' in self.tx_df_name:
            similarity = sample.similarity_std
            sentiment_md = sample.sentiment_md_std
            sentiment_q = sample.sentiment_q_std
            sentiment_a = sample.sentiment_a_std
            return (similarity, sentiment_md, sentiment_q, sentiment_a)

    def _get_financial_ratios(self, sample):
        '''
        This function extract the following financial ratios
        from the given data sample point
        '''
        if 'tx_v1' in self.tx_df_name:
            sue = sample.sue3_stand
            alpha = sample.alpha_stand
            volatility = sample.vol_m30_m1_stand
            mcap = sample.mcap_stand
            bm = sample.bm_stand
            roa = sample.roa_stand
            debt_assets = sample.debt_asset_stand
            smedest = sample.smedest_stand
            numest = sample.numest_stand
            sstdest = sample.sstdest_stand
            car_m1_m1 = sample.car_m1_m1_stand
            car_m2_m2 = sample.car_m2_m2_stand
            car_m30_m3 = sample.car_m30_m3_stand
            volume = sample.volume_stand
            return (alpha, car_m1_m1, car_m2_m2, car_m30_m3, sue, smedest,
                    numest, sstdest, mcap, roa, bm, debt_assets, volatility, volume)
        elif 'tx_v2' in self.tx_df_name:

            sue = sample.sue3_std
            volatility = sample.vol_call_m21_m1_std
            mcap = sample.mcap_std
            bm = sample.bm_std
            roa = sample.roa_std
            debt_assets = sample.debt_assets_std
            medest = sample.medest_std
            numest = sample.numest_std
            stdest = sample.stdest_std
            turnover = sample.turnover_ma21_std
            volume = sample.volume_ma21_std

            
            # if using factor benchmark, include alpha
            if 'car_ff3_call' in self.tasks[0]:
                car_m1_m1 = sample.car_ff3_call_m1_m1_std
                car_m2_m2 = sample.car_ff3_call_m2_m2_std
                car_m21_m3 = sample.car_ff3_call_m21_m3_std
                alpha = sample.alpha_ff3_call_std

                return (car_m1_m1, car_m2_m2, car_m21_m3, sue, medest,
                        numest, stdest, mcap, roa, bm, debt_assets, volatility, turnover, volume, alpha)

            else:
                ar_type = self.tasks[0].split('_')[0]
                ar_model = self.tasks[0].split('_')[1]

                # if the primary task is not CAR or BHAR, use CAR as default
                if ar_type not in ['car', 'bhar']:
                    ar_type = 'car'
                    ar_model = 'c5'

                car_m1_m1 = sample[f'{ar_type}_{ar_model}_call_m1_m1_std']
                car_m2_m2 = sample[f'{ar_type}_{ar_model}_call_m2_m2_std']
                car_m21_m3 = sample[f'{ar_type}_{ar_model}_call_m21_m3_std']

                return (car_m1_m1, car_m2_m2, car_m21_m3, sue, medest,
                        numest, stdest, mcap, roa, bm, debt_assets, volatility, turnover, volume)



    def _get_tasks(self, sample):
        '''
        Given sample, extract the target variables
        '''
        return [sample[t] for t in self.tasks]


class FrDataset(BaseDataset):
    '''A dataset with ONLY financial ratios

    Override the `__getitem__` method of BaseDataset

    Args:
        tx_df: DataFrame of CARs, financial ratios and manual text features. 
            Does not include text

        tasks: list of str. The name of target variables

        split_id: a string that indicates the start/end date of the the train/val/test datasets.
            e.g., '08/20-21' (train: 2008-2020, test: 2021)

        split_type: "train_val" or "test." A string indicating if the output is train_val or test

        split_df: DataFrame that keeps the split of windows

        split_type: str. 'train', 'val', or 'test'
    '''

    def __init__(self, split_type, split_id, tx_df_name, split_df_name, tasks, data_dir, **kwargs):

        # initialize "self.tx_df" and "self.split_df"
        super().__init__(split_type=split_type,
                         split_id=split_id,
                         tasks=tasks,
                         tx_df_name=tx_df_name,
                         split_df_name=split_df_name,
                         data_dir=data_dir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.tx_df.iloc[idx]

        # docid_idx is an int that's unique to each docid
        docid_idx = sample.docid_idx

        # get target variables
        t = self._get_tasks(sample)  # list

        # get financial ratios
        finratios = self._get_financial_ratios(sample)

        # get manual text features
        mantxts = self._get_manual_texts(sample)

        # get auxcar
        auxcars = self._get_auxcars(sample)

        # get inflow
        fund = self._get_fund(sample)

        # get revision
        revision = self._get_revision(sample)

        # get retail
        retail = self._get_retail(sample)

        # get auxvol
        auxvols = self._get_auxvols(sample)

        return (docid_idx,
                t,
                mantxts, finratios,
                auxcars, auxvols,
                fund, revision, retail)


class FrTxtDataset(BaseDataset):
    '''A dataset with BOTH financial ratios and texts

    Override the `__getitem__` method of BaseDataset

    Compared with FrDataset, FrTxtDataset has two new helper:
        _get_doctxt: Given a docid_idx, return a str of the call transcript.
        _data_integrity_check: make sure that every transcriptid in yx_df has NON-EMPTY text records in text_df.
            Otherwise, remove the transcriptid (row) in fr_df

    FrTxtDataset only return raw texts. Tokenization is done in `collate_fn`
    '''

    def __init__(self, split_type, split_id, d_model, tx_df_name, split_df_name,
                 preemb_dir, tasks, dataset_txt_return_type, data_dir, **kwargs):
        '''
        Compared with FrDataset, FrTxtDataset add one additional state:
            text_df: pandas DataFrame. Each row is a sentence.

        Args:
            (new) text_df: pands DataFrame. Each row is a sentence

            tasks: list of str. The name of target variables

            tx_df: DataFrame of CARs, financial ratios and manual text features. 
                Does not include text

            tasks: list of str. The name of target variables

            split_id: a string that indicates the start/end date of the the train/val/test datasets.
                e.g., '08/20-21' (train: 2008-2020, test: 2021)

            split_type: "train_val" or "test." A string indicating if the output is train_val or test

            split_df: DataFrame that keeps the split of windows

            split_type: str. 'train', 'val', or 'test'

        '''

        # initialize "self.tx_df" and "self.split_df"
        super().__init__(split_type=split_type,
                         tasks=tasks,
                         tx_df_name=tx_df_name,
                         split_df_name=split_df_name,
                         split_id=split_id,
                         data_dir=data_dir)

        # initialize "self.sent_metadata"
        # "sent_metadata" contains (tid, sid, section_type) pairs
        if 'tx_v1' in tx_df_name:
            self.dt_sents = read_feather(f'{data_dir}/f_sents_noempty_v1.feather', columns=['transcriptid', 'sentenceid', 'section_type'])
        elif 'tx_v2' in tx_df_name:
            self.dt_sents = read_feather(
                f'{data_dir}/f_dt_sents_metadata.feather')
        
        self.dt_sents.set_index('transcriptid', inplace=True)

        # save states
        self.dataset_txt_return_type = dataset_txt_return_type
        self.preemb_dir = preemb_dir
        self.d_model = d_model

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.tx_df.iloc[idx]

        # docid_idx is an int that's unique to eaqch docid
        docid_idx = sample.docid_idx

        # get targets
        t = self._get_tasks(sample)  # list

        # get financial ratios
        finratios = self._get_financial_ratios(sample)

        # get manual text features
        mantxts = self._get_manual_texts(sample)

        # get auxcar
        auxcars = self._get_auxcars(sample)

        # get inflow
        fund = self._get_fund(sample)

        # get revision
        revision = self._get_revision(sample)

        # get retail
        retail = self._get_retail(sample)

        # get auxvol
        auxvols = self._get_auxvols(sample)

        # get text as pre-embedding
        doc = self._get_doc(sample)

        return (docid_idx,
                t,
                mantxts, finratios,
                auxcars, auxvols,
                fund, revision, retail,
                doc)

    def _get_doc(self, sample):
        '''
        Return:
            doc: Dict[Dict[str, torch.Tensor]]
        '''
        tid = sample.transcriptid

        # for preemb
        dt_sents_sample = self.dt_sents.loc[tid]

        # load pre-embedding
        if 'tx_v1' in self.tx_df_name:
            preemb = torch.load(f'{self.preemb_dir}/{tid}.pt')
        elif 'tx_v2' in self.tx_df_name:
            year = sample.rdq.year
            preemb = torch.load(f'{self.preemb_dir}/{year}/{tid}.pt')

        # group preemb by each section_type
        doc = {
            'md': torch.zeros(1, self.d_model),
            'qa': torch.zeros(1, self.d_model)
        }

        for section_type in [['md'], ['q', 'a']]:
            sids = dt_sents_sample[dt_sents_sample.section_type.isin(
                section_type)].sentenceid

            # when len(sids)==0, it means the given section_type has null sentences
            # and an empty zero embedding is returned
            if len(sids) == 0:
                continue

            # if sid is a number, wrap it in a list,
            if isinstance(sids, np.int64):
                sids = [sids]
            # otherwise, convert sids to a list
            else:
                sids = sids.to_list()

            sids = sorted(sids)

            # select preemb with given section_type
            docemb = [preemb[s] for s in sids if s in preemb]
            if len(docemb) == 0:
                docemb = torch.zeros(1, self.d_model)
            else:
                docemb = torch.stack(docemb)  # (S,E)

            # add preemb to output
            if 'md' in section_type:
                doc['md'] = docemb
            else:
                doc['qa'] = docemb

        return doc


class GradPerpDemoDataset(torch.utils.data.Dataset):
    '''Toy dataset for illustration of GradPerp
    '''

    def __init__(self, tasks):
        # prepare data
        n_obs = 1000
        x = (torch.rand(n_obs)-0.5)*20
        self.x = x.unsqueeze(dim=-1)
        # y = (torch.rand(n_obs)-0.5)*20
        # self.y = y.unsqueeze(dim=-1)

        t_pri = self.f(x)
        t_aux1 = self.f_aux1(x)
        t_aux2 = self.f_aux2(x)

        t = {'pri': t_pri, 'aux1': t_aux1, 'aux2': t_aux2}
        t = torch.stack([t[task] for task in tasks], dim=1)

        self.t = t

    def __len__(self):
        return self.t.shape[0]

    def __getitem__(self, idx):
        return {'x': self.x[idx, :], 't': self.t[idx, :]}

    def f(self, x):
        return torch.sin(x)
        # return torch.sigmoid(x)

    def f_aux1(self, x):
        return torch.tanh(x)
        # return torch.sigmoid(x)

    def f_aux2(self, x):
        return torch.cos(x)
        # return torch.sigmoid(x)

