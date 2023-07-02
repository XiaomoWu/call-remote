import hydra
import numpy as np
import math
import pandas as pd
import lightning.pytorch as pl

import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchmetrics
import wandb

from datetime import datetime
from pathlib import Path
from pyarrow import feather
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from transformers import AutoModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        angular_speed = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        # even dimensions
        pe[:, 0::2] = torch.sin(position * angular_speed)
        # odd dimensions
        pe[:, 1::2] = torch.cos(position * angular_speed)

        self.register_buffer("pe", pe.unsqueeze(0))

        # normalize the final output
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x is N, L, D 19
        # pe is 1, maxlen, D
        scaled_x = self.layer_norm_1(x) * np.sqrt(self.d_model)
        encoded = scaled_x + self.pe[:, : x.size(1), :]
        return self.layer_norm_2(encoded)
        # return encoded


class FrModel(torch.nn.Module):
    """A non-text MTL model.

    It will be initialized as "self.model" in the MtlModel
    """

    def __init__(
        self,
        use_finratios,
        use_mantxts,
        dropouts,
        hidden_sizes,
        n_tasks,
        weighting_method_name,
        n_finratios=14,
        n_mantxts=4,
        **kwargs,
    ):
        super().__init__()

        # get n_nontxt_features
        n_nontxt_features = self.get_n_nontxt_features(
            use_finratios, use_mantxts, n_finratios, n_mantxts
        )

        # convert str hidden_size to int
        for i, s in enumerate(hidden_sizes):
            s = s.replace("in_dim", f"{n_nontxt_features}").replace(
                "out_dim", f"{n_tasks}"
            )
            try:
                s = eval(s)
                hidden_sizes[i] = int(s)
            except Exception:
                print(f"Unknown hidden size: {hidden_sizes}")

        # construct the hidden layers
        self.hidden_layers = nn.Sequential(
            *[
                self.get_fc_block(drop, in_dim, out_dim)
                for drop, in_dim, out_dim in zip(
                    dropouts[:-1], hidden_sizes, hidden_sizes[1:]
                )
            ]
        )

        # the final output fc layer
        self.fc_output = nn.Sequential(
            nn.Dropout(dropouts[-1]), nn.Linear(hidden_sizes[-1], n_tasks)
        )

        # save states
        self.use_finratios = use_finratios
        self.use_mantxts = use_mantxts
        self.n_finratios = n_finratios
        self.n_mantxts = n_mantxts

        # save states
        self.weighting_method_name = weighting_method_name

    def forward(self, batch):
        finratios = batch["finratios"]
        mantxts = batch["mantxts"]

        # feature fusion (for MTL)
        if self.use_mantxts and (not self.use_finratios):
            x = mantxts
        elif (not self.use_mantxts) and self.use_finratios:
            x = finratios
        elif self.use_mantxts and self.use_mantxts:
            x = torch.cat([finratios, mantxts], dim=1)

        # make prediction
        x = self.hidden_layers(x)
        y = self.fc_output(x)

        return y

    def get_fc_block(self, dropout, in_dim, out_dim):
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, out_dim), nn.ReLU())

    def get_shared_params(self):
        """All shared parameters (GradCos, GradPerp)"""
        shared_params = (
            [p for n, p in self.named_parameters() if "fc_output" not in n]
            if self.weighting_method_name in ["GradCos", "GradPerp", "GradNorm"]
            else None
        )

        return shared_params

    def get_task_specific_params(self):
        """Task-specific params (AdaMT)"""
        task_specific_params = None

        if self.weighting_method_name in ["AdaMt"]:
            task_specific_params = list(self.fc_output[-1].parameters())

        return task_specific_params

    def get_n_nontxt_features(self, use_finratios, use_mantxts, n_finratios, n_mantxts):
        n_nontxt_features = 0
        if use_finratios:
            n_nontxt_features += n_finratios
        if use_mantxts:
            n_nontxt_features += n_mantxts
        return n_nontxt_features


class FrTxtModel(torch.nn.Module):
    """A non-text MTL model.

    It will be initialized as "self.model" in MtlModel
    """

    def __init__(
        self,
        d_model,
        doc_encoding_pooling_method,
        dropout,
        use_finratios,
        use_mantxts,
        use_auxcars,
        use_auxvols,
        use_fund,
        use_revision,
        use_retail,
        expand_wide_features,
        n_tasks,
        weighting_method_name,
        datamodule_cfg,
        **kwargs,
    ):
        """
        Args:
            pretrained_sent_encoder_name: name of pre-trained model
            summarize_method: str
                cls: use the first [CLS] token as the summary
                mean: take the mean of every tokens (including the special tokens) as the summary
        """
        super().__init__()

        datamodule_txt_return_type = datamodule_cfg["datamodule_txt_return_type"]

        # get n_wide_features
        n_wide_features = self.get_n_wide_features(
            use_finratios,
            use_mantxts,
            use_auxcars,
            use_auxvols,
            use_fund,
            use_revision,
            use_retail,
            datamodule_cfg,
        )

        # get d_model
        self.d_model = d_model

        # init type_token embeddings
        # disabled, because we use different encoders for md and qa
        # 0: pad, 1: md, 2: qa
        # self.type_token_emb = nn.Embedding(
        #     3, embedding_dim=d_model, padding_idx=0)

        # ----------------------
        # init position encoding
        # ----------------------
        self.pe_md = PositionalEncoding(d_model=self.d_model)
        self.pe_qa = PositionalEncoding(d_model=self.d_model)

        # learnable position encoding (not used)
        # self.pos_md = nn.Parameter(torch.Tensor(512, self.d_model))  # (512,D)
        # self.pos_md = nn.Parameter(torch.Tensor(512, self.d_model))  # (512,D)

        # -----------------
        # init doc encoder
        # -----------------
        self.doc_encoder_md = self.init_doc_encoder(
            self.d_model, doc_encoding_pooling_method
        )
        self.doc_encoder_qa = self.init_doc_encoder(
            self.d_model, doc_encoding_pooling_method
        )

        # -------------------------------
        # init the final output fc layers
        # -------------------------------

        # 1) not using any wide featuers (deep text features only)
        # set n_enhanced_features to 0
        if (use_finratios is False) & (use_mantxts is False):
            n_enhanced_wide_features = 0

        # 1) using wide features and 2) also using wide feature enhancer
        elif expand_wide_features:
            n_enhanced_wide_features = 256

            self.fc_expand_wide_features = nn.Sequential(
                nn.Linear(n_wide_features, n_enhanced_wide_features), nn.ReLU()
            )
            self.fc_wide_features = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(n_enhanced_wide_features, n_enhanced_wide_features),
                nn.ReLU(),
            )

        # 1) using wide features but 2) NOT using wide feature enhancer
        else:
            n_enhanced_wide_features = n_wide_features

        # init fc_hidden
        self.fc_hidden = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                n_enhanced_wide_features + 2 * self.d_model,
                n_enhanced_wide_features + 2 * self.d_model,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                n_enhanced_wide_features + 2 * self.d_model,
                n_enhanced_wide_features + 2 * self.d_model,
            ),
            nn.ReLU(),
        )

        self.fc_output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_enhanced_wide_features + 2 * self.d_model, n_tasks),
        )

        # save other states
        self.use_finratios = use_finratios
        self.use_mantxts = use_mantxts
        self.use_auxcars = use_auxcars
        self.use_auxvols = use_auxvols
        self.use_fund = use_fund
        self.use_revision = use_revision
        self.use_retail = use_retail
        self.datamodule_cfg = datamodule_cfg
        self.expand_wide_features = expand_wide_features
        self.datamodule_txt_return_type = datamodule_txt_return_type
        self.weighting_method_name = weighting_method_name
        self.doc_encoding_pooling_method = doc_encoding_pooling_method
        self.n_enhanced_wide_features = n_enhanced_wide_features

        # init weights
        # self._init_weights()

    def _init_weights(self):
        # for module in self.modules():
        #     if isinstance(module, nn.Linear):
        #         # init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        #         init.trunc_normal_(module.weight, std=0.02)

        init.trunc_normal_(self.pos_md, std=0.02)
        init.trunc_normal_(self.pos_qa, std=0.02)

    def forward(self, batch):
        finratios = batch.get("finratios")
        mantxts = batch.get("mantxts")
        auxcars = batch.get("auxcars")
        auxvols = batch.get("auxvols")
        fund = batch.get("fund")
        revision = batch.get("revision")
        retail = batch.get("retail")

        doc_preembs = batch.get("doc_preembs")

        # -------------------------
        # level 1: get sentence emb
        # -------------------------

        sent_embs_md = doc_preembs["md"]["input_embeddings"]  # (B, L, D)
        sent_embs_qa = doc_preembs["qa"]["input_embeddings"]  # (B, L, D)

        attention_mask_md = doc_preembs["md"].get("attention_mask")  # (B, L)
        attention_mask_qa = doc_preembs["qa"].get("attention_mask")  # (B, L)

        # disabled, since we use different encoders for md and qa
        # sent_typeemb_md = self.type_token_emb(doc_typetokens['md'])
        # sent_typeemb_qa = self.type_token_emb(doc_typetokens['qa'])

        # -------------------------
        # level 2: get document emb
        # -------------------------

        doc_emb_md, doc_emb_qa = self.get_doc_emb(
            sent_embs_md,
            attention_mask_md,
            sent_embs_qa,
            attention_mask_qa,
            # sent_typeemb_md, sent_typeemb_qa,
            self.doc_encoding_pooling_method,
        )  # (B, D)

        # -------------------------
        # level 3: predictor
        # -------------------------

        # concate doc_type1 and doc_type3
        doc_emb = torch.cat([doc_emb_md, doc_emb_qa], dim=1)  # (B, 2D)

        # get wide_features (finratios/mantxts) if any
        wide_features = self.get_wide_features(
            finratios, mantxts, auxcars, auxvols, fund, revision, retail
        )  # (B, n_wide_features)

        # set final_features to doc_emb if no wide_features
        if wide_features is None:
            final_features = doc_emb

        # feature enhancer:
        # 1) expand wide features to 64; 2) residual connection
        else:
            if self.expand_wide_features:
                wide_features = self.fc_expand_wide_features(wide_features)

                wide_features = wide_features + self.fc_wide_features(wide_features)

            # MLP fusion
            final_features = torch.cat(
                [doc_emb, wide_features], dim=1
            )  # (B, 2D+wide_features)

        # final fc layers
        x = self.fc_hidden(final_features)  # (B, 2D+[wide_features])
        x = x + final_features  # (B, 2D+[wide_features])
        y = self.fc_output(x)  # (B, n_tasks)

        return y

    def pool_tsfm_output(
        self, seq_embs, pooling_method, attention_mask: torch.bool = None
    ):
        """Summarize token embeddings in a sequence into one
        Args:
            seq_emb: (B, L, D)
            attention_mask: (B, L). Binary, True to mask, False to keep
        """
        if pooling_method == "cls":
            seq_embs = seq_embs[:, 0, :]  # [N, E]

        elif pooling_method == "avg":
            # convert mask from bool to int
            # you must "flip" the mask using "~"!
            attention_mask = (
                ~attention_mask
            ).int()  # (B, L), int. 1 to keep, 0 to mask

            # expand the attention_mask so it has the same shape as seq_embs
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(seq_embs.size()).half()
            )  # (B, L, D), int. 1 to keep, 0 to mask

            seq_embs = torch.sum(seq_embs * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )  # (B, D)

            # (IMPORTANT) layer norm on the final output
            seq_embs = F.layer_norm(seq_embs, normalized_shape=(self.d_model,))

        return seq_embs

    def get_wide_features(
        self, finratios, mantxts, auxcars, auxvols, fund, revision, retail
    ):
        wide_features = []

        # add finratios/mantxts
        if self.use_finratios:
            wide_features.append(finratios)
        if self.use_mantxts:
            wide_features.append(mantxts)
        if self.use_auxcars:
            wide_features.append(auxcars)
        if self.use_auxvols:
            wide_features.append(auxvols)
        if self.use_fund:
            wide_features.append(fund)
        if self.use_revision:
            wide_features.append(revision)
        if self.use_retail:
            wide_features.append(retail)

        # return wide features conditionally
        if len(wide_features) == 0:
            return
        else:
            return torch.cat(wide_features, dim=1)

    def get_doc_emb(
        self,
        sent_embs_md,
        attention_mask_md,
        sent_embs_qa,
        attention_mask_qa,
        # sent_typeemb_md, sent_typeemb_qa,
        doc_encoding_pooling_method,
    ):
        """
        Args:
            sent_embs: List or Tensor
        Shape:
            sent_embs: (N,S,E) if Tensor; (N,) if List
        """

        # encode with transformers
        if "transformer" in doc_encoding_pooling_method:
            pooling_method = doc_encoding_pooling_method.split("_")[1]

            # layernorm on input embeddings (hurt performance)
            sent_embs_md = F.layer_norm(sent_embs_md, normalized_shape=(self.d_model,))
            sent_embs_qa = F.layer_norm(sent_embs_qa, normalized_shape=(self.d_model,))

            # add position encoding
            sent_embs_md = self.pe_md(sent_embs_md)
            sent_embs_qa = self.pe_qa(sent_embs_qa)

            # # add the "B" dimension to POS
            # bsz = sent_embs_md.size(0)
            # pos_md = self.pos_md.expand(bsz, -1, -1)
            # pos_qa = self.pos_qa.expand(bsz, -1, -1)

            # sent_embs_md = apply_pos(sent_embs_md, pos_md)
            # sent_embs_qa = apply_pos(sent_embs_qa, pos_qa)

            """
            # add type embedding (disabled since it hurts performance)
            sent_embs_md += sent_typeemb_md
            sent_embs_qa += sent_typeemb_qa
            """

            # encode
            doc_embs_md = self.doc_encoder_md(
                sent_embs_md, src_key_padding_mask=attention_mask_md
            )
            doc_embs_qa = self.doc_encoder_qa(
                sent_embs_qa, src_key_padding_mask=attention_mask_qa
            )

            # pool output
            doc_embs_md = self.pool_tsfm_output(
                doc_embs_md,
                pooling_method=pooling_method,
                attention_mask=attention_mask_md,
            )
            doc_embs_qa = self.pool_tsfm_output(
                doc_embs_qa,
                pooling_method=pooling_method,
                attention_mask=attention_mask_qa,
            )

        elif "gru" in doc_encoding_pooling_method:
            # encode
            doc_embs_md = self.doc_encoder_md(sent_embs_md)[0]
            doc_embs_qa = self.doc_encoder_md(sent_embs_qa)[0]

            # unpack output
            doc_embs_md, seq_lens_md = torch.nn.utils.rnn.pad_packed_sequence(
                doc_embs_md, batch_first=True
            )

            doc_embs_qa, seq_lens_qa = torch.nn.utils.rnn.pad_packed_sequence(
                doc_embs_qa, batch_first=True
            )

            # get the last output
            seq_idx_md = torch.arange(seq_lens_md.size(0))
            seq_idx_qa = torch.arange(seq_lens_qa.size(0))

            doc_embs_md = doc_embs_md[seq_idx_md, seq_lens_md - 1]
            doc_embs_qa = doc_embs_qa[seq_idx_qa, seq_lens_qa - 1]

        return doc_embs_md, doc_embs_qa

    def get_fc_block(self, dropout, in_dim, out_dim):
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, out_dim), nn.ReLU())

    def get_shared_params(self):
        """Get **last-layer** shared params

        We only get the last (few) layer(s) because text model has so many params that using all shared params is impossible
        """
        shared_params = (
            [
                p
                for n, p in self.named_parameters()
                if p.requires_grad and (("fc_hidden" in n) or ("fc_wide" in n))
            ]
            if self.weighting_method_name
            in ["GradNorm", "GradCos", "GradPerp", "OlAux"]
            else None
        )

        return shared_params

    def get_task_specific_params(self):
        """Task-specific params (AdaMT)"""
        task_specific_params = None

        if self.weighting_method_name in ["AdaMt"]:
            task_specific_params = list(self.fc_output[-1].parameters())

        return task_specific_params

    def get_n_wide_features(
        self,
        use_finratios,
        use_mantxts,
        use_auxcars,
        use_auxvols,
        use_inflow,
        use_revision,
        use_retail,
        datamodule_cfg,
    ):
        n_wide_features = 0
        if use_finratios:
            if "_ff3_" in datamodule_cfg.tasks[0]:
                # _ff3_ has "alpha"
                n_wide_features += 15
            else:
                n_wide_features += 14
        if use_mantxts:
            n_wide_features += 4
        if use_auxcars:
            n_wide_features += 3
        if use_auxvols:
            n_wide_features += 3
        if use_inflow:
            n_wide_features += 1
        if use_revision:
            n_wide_features += 1
        if use_retail:
            n_wide_features += 1
        return n_wide_features

    def init_doc_encoder(self, d_model, doc_encoding_pooling_method):
        if "transformer" in doc_encoding_pooling_method:
            doc_encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead=12, dim_feedforward=1024, dropout=0.3, batch_first=True
            )

            doc_encoder = nn.TransformerEncoder(
                doc_encoder_layer, num_layers=3, enable_nested_tensor=True
            )

        elif "gru" in doc_encoding_pooling_method:
            doc_encoder = nn.GRU(
                input_size=d_model,
                hidden_size=int(d_model / 2),
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )

        else:
            raise Exception(f"Unknown doc_encoding_pooling_method")

        return doc_encoder

    def init_sent_encoder(
        self,
        dataset_txt_return_type,
        pretrained_sent_encoder_name,
        unfreeze_sent_encoder_embedding,
        n_unfreezed_layers,
    ):
        # Don't init sent_encoder if return preemb
        if dataset_txt_return_type == "preemb":
            return

        # otherwise, init sent_encoder
        sent_encoder = AutoModel.from_pretrained(pretrained_sent_encoder_name)
        sent_encoder.train()
        print(f"sent_encoder initialized!")

        # freeze all params
        for p in sent_encoder.parameters():
            p.requires_grad = False

        # unfreeze selectively
        if "sentence-transformers" in pretrained_sent_encoder_name:
            sent_encoder.eval()
            return sent_encoder

        else:
            # unfreeze embeddings
            if unfreeze_sent_encoder_embedding == True:
                for p in sent_encoder.embeddings.parameters():
                    p.requires_grad = True

            # unfreeze last N transformer layers
            if n_unfreezed_layers == "all":
                for p in sent_encoder.transformer.layer.parameters():
                    p.requires_grad = True
                return sent_encoder

            elif n_unfreezed_layers == 0:
                return sent_encoder

            elif n_unfreezed_layers > 0:
                layer_ids = sorted(
                    list(
                        set(
                            n[0]
                            for n, p in sent_encoder.transformer.layer.named_parameters()
                        )
                    )
                )  # List[Int]

                unfreezed_layers = layer_ids[-n_unfreezed_layers:]

                for n, p in sent_encoder.transformer.layer.named_parameters():
                    if n[0] in unfreezed_layers:
                        p.requires_grad = True

                return sent_encoder


class FrTxtMQModel(FrTxtModel):
    def __init__(
        self,
        d_model,
        doc_encoding_pooling_method,
        dropout,
        use_finratios,
        use_mantxts,
        use_auxcars,
        use_auxvols,
        use_fund,
        use_revision,
        use_retail,
        expand_wide_features,
        n_tasks,
        weighting_method_name,
        datamodule_cfg,
        mq_enc_dec_times=1,
        qsz=128,  # length of task-specific queries, default 48
        **kwargs,
    ):
        """Args:
        mq_enc_dec_times: int. Layers of MQ encoder and decoder. Should be 1.
        """
        super().__init__(
            d_model,
            doc_encoding_pooling_method,
            dropout,
            use_finratios,
            use_mantxts,
            use_auxcars,
            use_auxvols,
            use_fund,
            use_revision,
            use_retail,
            expand_wide_features,
            n_tasks,
            weighting_method_name,
            datamodule_cfg,
            **kwargs,
        )

        # del layers created by FrTxtModel but not used in FrTxtMQModel
        del (
            self.pe_md,
            self.pe_qa,
            self.doc_encoder_md,
            self.doc_encoder_qa,
            self.fc_hidden,
        )

        # get n_wide_features
        n_wide_features = self.get_n_wide_features(
            use_finratios,
            use_mantxts,
            use_auxcars,
            use_auxvols,
            use_fund,
            use_revision,
            use_retail,
            datamodule_cfg,
        )

        # linear projection to create task-specific features
        self.linears_md = nn.ModuleList(
            [nn.Linear(d_model, d_model) for i in range(n_tasks)]
        )
        self.linears_qa = nn.ModuleList(
            [nn.Linear(d_model, d_model) for i in range(n_tasks)]
        )

        # MQTransformer Encoder-Decoders
        self.enc_dec_md_list = nn.ModuleList(
            [MQEncDec(n_tasks, d_model, qsz) for i in range(mq_enc_dec_times)]
        )
        self.enc_dec_qa_list = nn.ModuleList(
            [MQEncDec(n_tasks, d_model, qsz) for i in range(mq_enc_dec_times)]
        )

        # final fc output layers
        # method 1: each task has its own fc_output
        self.fc_output = nn.ModuleList(
            [
                nn.Linear(2 * d_model + self.n_enhanced_wide_features, 1)
                for i in range(n_tasks)
            ]
        )

        # method 2: all tasks share the same fc_output
        # self.fc_output = nn.Linear(
        #     n_tasks * (2 * d_model + self.n_enhanced_wide_features), n_tasks
        # )

        # A learnable "M" for GradPerp (may not be used)
        self.M = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        # get inputs
        finratios = batch.get("finratios")
        mantxts = batch.get("mantxts")
        auxcars = batch.get("auxcars")
        auxvols = batch.get("auxvols")
        fund = batch.get("fund")
        revision = batch.get("revision")
        retail = batch.get("retail")

        doc_preembs = batch.get("doc_preembs")

        # -------------------------
        # level 1: get sentence emb
        # -------------------------

        sent_embs_md = doc_preembs["md"]["input_embeddings"]  # (B, L, D)
        sent_embs_qa = doc_preembs["qa"]["input_embeddings"]  # (B, L, D)

        attention_mask_md = doc_preembs["md"].get("attention_mask")  # (B, L)
        attention_mask_qa = doc_preembs["qa"].get("attention_mask")  # (B, L)

        # task specific embeddings
        sent_embs_md_list = []  # list of (B, L, D)
        for i, linear in enumerate(self.linears_md):
            x = linear(sent_embs_md)
            sent_embs_md_list.append(x)

        sent_embs_qa_list = []  # list of (B, L, D)
        for i, linear in enumerate(self.linears_qa):
            x = linear(sent_embs_qa)
            sent_embs_qa_list.append(x)

        # -------------------------
        # level 2: get document emb
        # -------------------------

        # MD: enc_dec
        out = sent_embs_md_list
        for i, enc_dec_md in enumerate(self.enc_dec_md_list):
            out = enc_dec_md(out, attention_mask_md)  # list of (B, L, D)

        # MD: pooling
        doc_emb_md = []  # list of (B, D)
        for x in out:
            doc_emb_md.append(self.pool_tsfm_output(x, "avg", attention_mask_md))

        # QA: enc_dec
        out = sent_embs_qa_list
        for i, enc_dec_qa in enumerate(self.enc_dec_qa_list):
            out = enc_dec_qa(out, attention_mask_qa)  # list of (B, L, D)

        # QA: pooling
        doc_emb_qa = []  # list of (B, D)
        for x in out:
            doc_emb_qa.append(self.pool_tsfm_output(x, "avg", attention_mask_qa))

        # -------------------------
        # level 3: predictor
        # -------------------------

        # concate doc_type1 and doc_type3
        doc_emb = [
            torch.cat([md, qa], dim=1) for md, qa in zip(doc_emb_md, doc_emb_qa)
        ]  # list of (B, 2D)

        # get wide_features (finratios/mantxts) if any
        wide_features = self.get_wide_features(
            finratios, mantxts, auxcars, auxvols, fund, revision, retail
        )  # (B, wide_features)

        # set final_features to doc_emb if no wide_features
        if wide_features is None:
            final_features = doc_emb

        # feature enhancer:
        # 1) expand wide features to 64; 2) residual connection
        else:
            if self.expand_wide_features:
                wide_features = self.fc_expand_wide_features(wide_features)

                wide_features = wide_features + self.fc_wide_features(
                    wide_features
                )  # (B, wide_features)

            # MLP fusion
            # list of (B, 2D+wide_features)
            final_features = [torch.cat([emb, wide_features], dim=1) for emb in doc_emb]

            # # concat the list of embeddings
            # final_features = torch.cat(final_features, dim=1)  # (B, n_task * (2D+wide_features))

        # final fc output layers
        # Method 1: each task has its own fc layer
        x = [
            fc(feat) for fc, feat in zip(self.fc_output, final_features)
        ]  # list of (B, 1)
        y = torch.cat(x, dim=1)  # (B, n_tasks)

        """
        # Method 2: shared fc layers
        final_features = torch.cat(
            final_features, dim=1
        )  # (B, n_task * (2D+wide_features))
        y = self.fc_output(final_features)  # (B, n_tasks)
        """

        return y

    def get_shared_params(self):
        """Get **last-layer** shared params

        We only get the last (few) layer(s) because text model has so many params that using all shared params is impossible
        """
        shared_params = (
            [
                param
                for name, param in self.named_parameters()
                if param.requires_grad
                and (
                    # method 1: each layer has its own fc layer
                    ("enc_dec_md_list.0.decoder.ffn" in name)
                    or ("enc_dec_qa_list.0.decoder.ffn" in name)
                    # method 2: shared fc layers
                    # "fc_output"
                    # in name
                )
            ]
            if self.weighting_method_name
            in ["GradNorm", "GradCos", "GradPerp", "OlAux"]
            else None
        )

        return shared_params

    def get_task_specific_params(self):
        """Task-specific params (AdaMT)"""
        task_specific_params = None

        if self.weighting_method_name in ["AdaMt"]:
            task_specific_params = list(self.fc_output.parameters())

        return task_specific_params


class MQEncDec(nn.Module):
    def __init__(self, n_tasks, d_model, qsz):
        """
        Args:
            qsz: query size, i.e., length of a query, the "N" in the paper
        """
        super(MQEncDec, self).__init__()

        self.n_tasks = n_tasks
        self.d_model = d_model
        self.qsz = qsz

        # init task queries and pos
        self.query = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(self.qsz, self.d_model))
                for i in range(self.n_tasks)
            ]
        )  # list of (qsz, D)
        self.query_pos = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(self.qsz, self.d_model))
                for i in range(self.n_tasks)
            ]
        )  # list of (qsz, D)  the "blue circle" in the paper
        self.feat_pos = nn.ParameterList(
            [nn.Parameter(torch.Tensor(512, self.d_model)) for i in range(self.n_tasks)]
        )  # list of (512, D) the "black circle" in the paper. We set 512 because the max length (num sentences) of a document is set to 512

        self.norm = nn.LayerNorm(self.d_model)

        # init encoders and decoders
        self.encoder = MQEncoder(d_model)
        self.decoder = MQDecoder(d_model)
        # self.act = nn.GELU()

        # init cross-task attention
        self.cross_encoder = nn.TransformerEncoderLayer(
            d_model,
            nhead=12,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        # init weights
        self._init_weights()

    def _init_weights(self):
        # # init encoder and decoder
        # init.trunc_normal_(self.encoder.weight, std=.02)
        # init.trunc_normal_(self.cross_encoder.weight, std=.02)
        # init.trunc_normal_(self.decoder.weight, std=.02)

        # init task queries and pos
        # In the paper, they init TWICE, WHY??
        for i in range(self.n_tasks):
            init.kaiming_uniform_(self.query[i], a=math.sqrt(5))
            init.trunc_normal_(self.query[i], std=0.02)

            init.kaiming_uniform_(self.query_pos[i], a=math.sqrt(5))
            init.trunc_normal_(self.query_pos[i], std=0.02)

            init.kaiming_uniform_(self.feat_pos[i], a=math.sqrt(5))
            init.trunc_normal_(self.feat_pos[i], std=0.02)

    def forward(self, x_list, attention_mask):
        """
        Args:
            x_list: task-specific sentence embeddings, list of (B, L, D)
            attention_mask: list of (B, L). Binary, True to mask (ignore), False to attend
        """

        # ------------------------
        # Add "B" dimension to POS
        # ------------------------
        bsz = x_list[0].shape[0]

        query_list = []  # list of (B, qsz, D)
        query_pos_list = []  # list of (B, qsz, D)
        feat_pos_list = []  # list of (B, 512, D)

        for i, p in enumerate(self.query):
            _x = p.expand(bsz, -1, -1)
            query_list.append(_x)

        for i, p in enumerate(self.query_pos):
            _x = p.expand(bsz, -1, -1)
            query_pos_list.append(_x)

        for i, p in enumerate(self.feat_pos):
            _x = p.expand(bsz, -1, -1)
            feat_pos_list.append(_x)

        # --------------
        # shared encoder
        # --------------
        query_for_dec_list = []  # list of (B, qsz, D)
        for x, query, query_pos, feat_pos in zip(
            x_list, query_list, query_pos_list, feat_pos_list
        ):
            _q = self.encoder(x, query, query_pos, feat_pos, attention_mask)
            query_for_dec_list.append(_q)

        # --------------------
        # cross-task attention
        # --------------------
        # (B, n_tasks * qsz, D)
        query_cat = torch.cat(query_for_dec_list, dim=1)
        query_cat = self.norm(query_cat)
        query_cat = self.cross_encoder(
            # ALL nan if use "gelu!"
            query_cat,
            src_key_padding_mask=None,
        )  # (B, n_tasks * qsz, D)

        query_cat = torch.split(query_cat, self.qsz, dim=1)  # list of (B, qsz, D)

        # --------------
        # shared decoder
        # --------------
        out_feature_list = []
        for x, query, query_pos, feat_pos in zip(
            x_list, query_cat, query_pos_list, feat_pos_list
        ):
            _f = self.decoder(x, query, query_pos, feat_pos, attention_mask)
            # _f= self.act(_f)  # MQT add this act layer but I don't know why
            out_feature_list.append(_f)

        return out_feature_list  #


class MQEncoder(nn.Module):
    def __init__(self, d_model):
        """MQ encoder"""
        super(MQEncoder, self).__init__()

        self.norm = nn.LayerNorm(d_model)
        self.enc_attn = nn.MultiheadAttention(
            d_model, num_heads=12, dropout=0.1, batch_first=True
        )
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)

        # the "query-learning" block in the shared encoder
        self.qlearn = MQQlearn(d_model)
        self.enc_ffn = MQMlp(d_model)

    def forward(self, x, query, query_pos, feat_pos, mask):
        """
        Args:
            x: (B, L, D)  task-specific features
            query: (B, qsz, D)  task-specific query

            kpos: (B, 512, D)
            qpos: (B, qsz, D)
            mask: (B, L)
        """

        # LN + MHA
        q = apply_pos(query, query_pos)  # (B, qsz, D)
        k = apply_pos(x, feat_pos)  # (B, L, D)
        v = apply_pos(x, feat_pos)  # (B, L, D)

        q, k, v = self.norm(query), self.norm(x), self.norm(x)
        attn_out = self.enc_attn(q, k, v, key_padding_mask=mask)[0]  # (B, qsz, D)
        query = query + self.drop1(attn_out)  # (B, qsz, D)

        query = self.qlearn(query)  # (B, qsz, D)

        # (B, qsz, D)
        query = query + self.drop2(self.enc_ffn(self.norm(query)))
        return query


def apply_pos(x, pos):
    """
    Args:
        x: (B, L, D)  (L<=512)
        pos: (B, 512, D)
    """

    x = x + pos[:, : x.shape[1], :]  # (B, L, D)
    return x


class MQDecoder(nn.Module):
    def __init__(self, d_model):
        """MQ decoder"""
        super(MQDecoder, self).__init__()

        self.mha = nn.MultiheadAttention(
            d_model, num_heads=12, dropout=0.1, batch_first=True
        )  # a MHA where q is task features and k, v are task queries

        self.norm = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.ffn = MQMlp(d_model)

    def forward(self, x, query, query_pos, feat_pos, mask=None):
        """
        Args:
            query: (B, qsz, D)  task-specific query
            x: (B, L, D)  task-specific features
            mask: None. Since "K" is query, we don't need mask
        """
        x = apply_pos(x, feat_pos)  # (B, L, D)
        query = apply_pos(query, query_pos)  # (B, qsz, D)

        # First decoder
        q, k, v = self.norm(x), self.norm(query), self.norm(query)
        feat = self.mha(q, k, v)[0]
        feat = x + self.drop1(feat)
        feat = feat + self.drop2(self.ffn(self.norm(feat)))

        return feat


class MQQlearn(nn.Module):
    """The "query learning" block in the shared encoder"""

    def __init__(self, d_model):
        super(MQQlearn, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, query):
        """
        Args:
            query: (B, qsz, D)
        """
        query1 = self.linear(self.norm(query))  # (B, qsz, D)
        return query + query1


class MQMlp(nn.Module):
    """The MLP block in a MQ encoder/decoder (residual connection is not included!)"""

    def __init__(self, d_model, hidden_features=2048, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GradPerpDemoModel(torch.nn.Module):
    def __init__(self, hidden_size, n_tasks, dropout=0.3, **kwargs):
        super().__init__()

        # disable auto backward
        self.automatic_optimization = False

        self.fc_hidden = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fc_output = nn.Linear(hidden_size, n_tasks)

    def forward(self, batch):
        x = batch["x"]

        y = self.fc_hidden(x)
        y = self.fc_output(y)

        return y

    def _get_shared_params(self):
        return list(self.fc_hidden.parameters())

    def get_task_specific_params(self):
        pass


class MtlModel(pl.LightningModule):
    """The one MTL model

    All models, no matte whether they're with or without text, or whatever kind of encoders (e.g., FrModel, FrTxtModel) they use, will be initizalized by it.

    `MtlModel` implement the lightning components
    """

    def __init__(
        self,
        model_cfg,
        datamodule_cfg,
        optimizer_cfg,
        scheduler_cfg,
        weighting_method_cfg,
        trainer_cfg,
        work_dir,
    ):
        super().__init__()
        self.save_hyperparameters()

        # disable auto backward
        self.automatic_optimization = False

        # get derived configs
        self.n_tasks = len(datamodule_cfg.tasks)
        self.weighting_method_name = weighting_method_cfg.name

        # ----------------------
        # init loss functions
        # ----------------------
        self.val_loss = torchmetrics.MeanSquaredError()
        self.val_ev = torchmetrics.ExplainedVariance()
        # self.val_acc = torchmetrics.Accuracy()
        self.test_ev = torchmetrics.functional.explained_variance

        # -------------------------
        # add additional parameters
        # -------------------------
        #   - if 'Uncert', add logvars
        #   - if 'GradNorm', create and add task_weights
        self.add_logvars_param()
        self.add_task_weights_param()

        # ----------------------
        # init weighting_method
        # ----------------------
        self.weighting_method = hydra.utils.instantiate(
            weighting_method_cfg[weighting_method_cfg.name],
            n_tasks=self.n_tasks,
            max_epochs=trainer_cfg.max_epochs,
            task_weights=self.task_weights,
        )

        # ----------------------
        # init forward model
        # ----------------------
        self.model = hydra.utils.instantiate(
            model_cfg,
            datamodule_cfg=datamodule_cfg,
            n_tasks=self.n_tasks,
            weighting_method_name=self.weighting_method_name,
            _recursive_=False,
        )

        # save states
        self.model_cfg = model_cfg
        self.datamodule_cfg = datamodule_cfg
        self.optimizer_cfg = optimizer_cfg
        self.weighting_method_cfg = weighting_method_cfg
        self.wdir = Path(work_dir)

        self.test_step_outputs = []  # new in v2.0

    # forward
    def forward(self, batch):
        docid_idx = batch["docid_idx"]
        y = self.model(batch)
        t = self.get_truth(batch)

        return docid_idx, y, t

    # training
    def training_step(self, batch, batch_idx):
        # get optimizer/scheduler
        opt = self.optimizers()
        sch = self.lr_schedulers()

        # make prediction
        y = self.model(batch)

        # compute loss
        t = self.get_truth(batch)
        losses = self.calc_losses(y, t)  # (n_task,)

        # backward
        self.weighting_method.backward(
            losses=losses,
            pl_module=self,
            shared_params=self.model.get_shared_params(),
            task_specific_params=self.model.get_task_specific_params(),
        )

        # step/accumulate grad every N batches
        if (batch_idx + 1) % self.model_cfg.manual_step_every_n_batches == 0:
            # step optimizer
            opt.step()
            opt.zero_grad()

            # step schduler
            if isinstance(sch, (LambdaLR,)):
                sch.step()

        # logging
        self.training_step_logging(losses)

    # Validation
    def validation_step(self, batch, idx):
        # make prediction
        y = self.model(batch)

        # compute loss
        t = self.get_truth(batch)
        self.val_loss(y[:, 0], t[:, 0])
        self.val_ev(y[:, 0], t[:, 0])

    def on_validation_epoch_end(self):
        # print task weight
        if hasattr(self.weighting_method, "task_weights"):
            print(f"task_weights={self.weighting_method.task_weights}")

        # only print "M" when it's learned
        if self.weighting_method_name == "GradPerp":
            if self.weighting_method_cfg.GradPerp.M == -1:
                print(f"M={self.model.M.detach()}")

        # log metrics
        val_ev = self.val_ev.compute() * 100
        self.log(
            "val/loss",
            self.val_loss.compute(),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log("val/EV", val_ev, sync_dist=True)

        # for debugging
        if self.global_rank == 0:
            print(f"val/EV={val_ev:.3f}%")

        self.val_ev.reset()
        self.val_loss.reset()

        # step scheduler
        sch = self.lr_schedulers()
        if isinstance(sch, (ReduceLROnPlateau,)):
            sch.step(val_ev)

    # test
    def test_step(self, batch, batch_idx):
        # make prediction
        y = self.model(batch)

        # get truth
        t = self.get_truth(batch)

        # collect outputs (new in v2.0)
        self.test_step_outputs.append({"docid_idx": batch["docid_idx"], "y": y, "t": t})

    def on_test_epoch_end(self):
        # collect yt
        outputs = self.test_step_outputs
        docid_idx, y, t = self.collect_yt(outputs)

        # save yt to DataFrame (and wandb)
        yt = self.save_yt(docid_idx, y, t)

        # log performance
        # - This must be performed after yt is created
        #   otherwise there might be duplicats in yt
        t_pri = self.hparams.datamodule_cfg.tasks[0]
        test_ev = (
            self.test_ev(torch.tensor(yt[f"y_{t_pri}"]), torch.tensor(yt[f"t_{t_pri}"]))
            * 100
        )

        # log performance
        if self.global_rank == 0:
            print(f"test/EV={test_ev:.3f}%")
        self.log("test/EV", test_ev, sync_dist=True)
        self.log("test/N_obs", float(yt.shape[0]), sync_dist=True)

    # optimizer
    def configure_optimizers(self):
        # get params
        params = self.get_opt_params(self.model_cfg, self.optimizer_cfg)

        # init optimizer
        optimizer = self.init_optimizer(self.hparams.optimizer_cfg, params)

        # init scheduler (if any)
        if self.hparams.scheduler_cfg is None:
            return optimizer

        else:
            scheduler = hydra.utils.instantiate(
                self.hparams.scheduler_cfg, optimizer=optimizer
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                },
            }

    # helpers
    def add_task_weights_param(self):
        """If GradNorm, create a param for task weights
        Note:
            In other weighting method, task_weights is created by the weighting_method instead.
        """
        self.task_weights = None
        if self.weighting_method_name == "GradNorm":
            task_weights = torch.ones(self.n_tasks, device=self.device) / self.n_tasks
            task_weights.requires_grad_()
            self.task_weights = nn.Parameter(task_weights)

    def add_logvars_param(self):
        """Assign sigmas if loss/name is sigma, otherwise assigned None"""
        self.logvars = None

        if self.weighting_method_name == "Uncert":
            self.logvars = nn.Parameter(torch.zeros(self.n_tasks), requires_grad=True)

    def calc_losses(self, y, t):
        losses = [F.mse_loss(y[:, i], t[:, i]) for i in range(self.n_tasks)]
        return torch.stack(losses)

    def collect_yt(self, outputs):
        """Collect yt and convert them to approprite dimension"""
        # collect yt from one process
        docid_idx = torch.cat([x["docid_idx"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        t = torch.cat([x["t"] for x in outputs])

        # all_gather yt
        docid_idx = self.all_gather(docid_idx).detach()
        y = self.all_gather(y).detach()
        t = self.all_gather(t).detach()
        assert docid_idx.dim() == 2
        assert y.dim() == t.dim() == 3
        assert y.size(-1) == t.size(-1) == self.n_tasks

        # convert yt dimension
        docid_idx = docid_idx.flatten().tolist()  # 1D
        y = y.reshape(-1, self.n_tasks)  # (N, n_tasks)
        t = t.reshape(-1, self.n_tasks)  # (N, n_tasks)

        return docid_idx, y, t

    def get_opt_params(self, model_cfg, optimizer_cfg):
        """Get trainable params for optimizer

        Return:
            params: List[Dict]
        """
        # get trainable params
        if isinstance(self.model, (FrModel, GradPerpDemoModel)):
            params = [p for p in self.parameters() if p.requires_grad]

        elif isinstance(self.model, FrTxtMQModel):
            params_M = [
                param
                for name, param in self.named_parameters()
                if param.requires_grad and name == "model.M"
            ]
            params_others = [
                param
                for name, param, in self.named_parameters()
                if param.requires_grad and name != "model.M"
            ]
            params = [
                {"params": params_M, "lr": model_cfg.get("lr_M")},
                {"params": params_others, "lr": optimizer_cfg.get("lr")},
            ]
            # params = [p for p in self.parameters() if p.requires_grad]

        elif isinstance(self.model, FrTxtModel):
            fc_lr = model_cfg.get("fc_lr")
            doc_encoder_lr = model_cfg.get("doc_encoder_lr")

            # learning rate sanity check
            assert fc_lr is not None, f"Require fr_lr not None!"

            # add FC and encoder params
            fc_params = [
                p for n, p in self.model.named_parameters() if n.startswith("fc_")
            ]
            doc_encoder_params = [
                p
                for n, p in self.model.named_parameters()
                if n.startswith("doc_encoder")
            ]
            params = [
                {"params": filter(lambda p: p.requires_grad, params), "lr": lr}
                for params, lr, in zip(
                    [fc_params, doc_encoder_params], [fc_lr, doc_encoder_lr]
                )
                if len(params) > 0
            ]

            # # add PE params
            # pe_params = [
            #     p for n, p in self.model.named_parameters()
            #     if n.startswith('pe_')
            # ]
            # params.append({'params': pe_params, 'lr': fc_lr})

            if self.weighting_method_name in ["GradNorm"]:
                params.append({"params": self.task_weights, "lr": 1e-3})

        else:
            raise Exception(f"Unknown model: {type(self)}")

        return params

    def get_phi_grads(self):
        """In the training step, get grads of the final linear layer (for AdaMt)"""
        # Note:
        #   only the following loss need this step:
        #   - AdaMt
        if self.hparams.weighting_method_cfg._target_.split(".")[-1] in ["AdaMt"]:
            # get w and b of the last layer
            w_grad, b_grad = [param.grad for param in self.fc_output.parameters()]

            # concat w and b
            if w_grad is not None:
                return torch.cat([w_grad, b_grad.unsqueeze(dim=1)], dim=1)

    def get_theta_grads(self):
        """In the training step, get grads of the final linear layer
        (the "Phi" in Du et al., 2018)
        """
        # Note:
        #   only the following loss need this step:
        #   - AdaMt
        if self.hparams.weighting_method_cfg._target_.split(".")[-1] in [
            "AdaMt",
            "GradCos",
        ]:
            theta_grads = torch.cat(
                [
                    param.grad.flatten()
                    for name, param in self.named_parameters()
                    if ("fc_output" not in name)
                ]
            )

            return theta_grads  # 1D vector

    def get_truth(self, batch):
        return batch["t"]

    def init_optimizer(self, optimizer_cfg, params):
        optimizer_name = optimizer_cfg._target_.split(".")[-1]

        # depends on whether the optimizer is DeepSpeed, initialize differently
        if "DeepSpeed" in optimizer_name or "HybridAdam" in optimizer_name:
            optimizer = hydra.utils.instantiate(
                optimizer_cfg, model_params=params, _convert_="partial"
            )
        else:
            optimizer = hydra.utils.instantiate(
                optimizer_cfg, params=params, _convert_="partial"
            )

        return optimizer

    def training_step_logging(self, losses):
        # log task_weights
        if hasattr(self.weighting_method, "task_weights"):
            for t, w in zip(
                self.hparams.datamodule_cfg.tasks, self.weighting_method.task_weights
            ):
                self.log(f"train/task_weight_{t}", torch.tensor(w).float())

        # only log "M" when it is trainable
        if self.weighting_method_name == "GradPerp":
            if self.weighting_method_cfg.GradPerp.M == -1:
                self.log(f"train/M", self.model.M.detach())

        # log total loss
        if hasattr(self.weighting_method, "tot_loss"):
            self.log(f"train/tot_loss", self.weighting_method.tot_loss)

        # log primary task loss
        self.log(f"train/pri_loss", losses[0])

    def save_yt(self, docid_idx, y, t) -> pd.DataFrame:
        # get split_id
        split_id = self.datamodule_cfg.split_id

        # self.logger._experiment.id ('heyp0n2n')
        # self.logger._experiment.name ('08q2-10q1/10q2')
        # self.logger._experiment.settings.timespec ('20230623_234341')

        # --------------------------------
        # collect results into a DataFrame
        # --------------------------------
        yt = {}
        yt["docid_idx"] = docid_idx
        for i, task in enumerate(self.hparams.datamodule_cfg.tasks):
            yt[f"y_{task}"] = y[:, i].tolist()
            yt[f"t_{task}"] = t[:, i].tolist()

        # get split_id
        yt["split_id"] = split_id

        # get pred_target and feature_set
        weighting_method = self.weighting_method_name.lower()
        pred_target = self.datamodule_cfg.tasks[0]
        feature_set = (
            "frtxt" if "FrTxtData" in self.datamodule_cfg["_target_"] else "fr"
        )
        yt["pred_target"] = pred_target
        yt["feature_set"] = feature_set

        # get pred_model
        if isinstance(self.model, FrModel):
            pred_model = "mlp???"
        elif isinstance(self.model, FrTxtModel):
            pred_model = "tsfm or gru"
        yt["pred_model"] = pred_model

        # get weighting method
        yt["weight_method"] = weighting_method
        if len(self.datamodule_cfg.tasks) == 1:
            yt["weight_method"] = "stl"

        # convert to DataFrame
        yt = pd.DataFrame(yt)
        yt = yt.drop_duplicates(subset=["docid_idx"])

        # ---------------------
        # save yt to local disk
        # ---------------------

        # only save when self.logger is a valid wandb logger
        # otherwise, we can't get valid sweep_id, run_id, etc.
        if (not isinstance(self.logger, pl.loggers.WandbLogger)) or (
            not hasattr(self.logger.experiment.settings, "timespec")
        ):
            return yt

        # create the folder
        split_id_new = (
            split_id.replace("/", "(") + ")"
        )  # 08q2-10q1/10q2 -> 08q2-10q1(10q2)
        sweep_id = self.logger.experiment.sweep_id
        run_id = self.logger.experiment.id
        run_timespec = self.logger.experiment.settings.timespec

        if sweep_id is None:
            savedir = (
                self.wdir / f"data/v2/eval/yt-temp/(run_id)-{run_timespec}-{run_id}"
            )
        else:
            savedir = self.wdir / f"data/v2/eval/yt-temp/(sweep_id)-{sweep_id}"

        """
        # run-specific, should be removed later
        if sweep_id is None:
            savedir = self.wdir / f'data/v2/eval/yt-temp/(run_id)-{run_timespec}-{run_id}/{lr}'
        else:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            savedir = self.wdir / f'data/v2/eval/yt-temp/(sweep_id)-{sweep_id}/{lr}'
        """

        savedir.mkdir(parents=True, exist_ok=True)  # create folder if not exist

        # create the file name
        savename = f"{split_id_new}-{run_timespec}-{run_id}(run_id)"

        # write to disk and upload
        feather.write_feather(yt, f"{savedir}/{savename}.feather")
        self.logger.experiment.log({"yt": wandb.Table(dataframe=yt)})

        return yt
