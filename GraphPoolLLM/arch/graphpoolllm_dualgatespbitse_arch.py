from math import sqrt

import torch
import torch.nn as nn
import numpy as np
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoTokenizer
import transformers
from .tokenizer_processor import TokenizerConfig
import torch.nn.functional as F
from .StandardNorm import Normalize

transformers.logging.set_verbosity_error()

class NodeSelector(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):  # x: [B, N, T]
        scores = x.mean(dim=-1).squeeze(-1)
        return scores

def gumbel_softmax_topk(scores, k, tau=1.0, eps=1e-10):
    # scores: [B, N]
    B, N = scores.size()
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + eps) + eps)
    logits = (scores + gumbel_noise) / tau
    probs = torch.softmax(logits, dim=-1)  # [B, N]

    topk_probs, topk_idx = torch.topk(probs, k, dim=-1)
    mask = torch.zeros_like(probs).scatter_(1, topk_idx, 1.0)
    return topk_idx, mask

def extract_subgraph(x, topk_idx):
    # x: [B, N, T], topk_idx: [B, K]
    B, N, T = x.size()
    x_sub = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(-1, -1, T))  # [B, K, T]
    return x_sub


class AttentionLayer(nn.Module):
 

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out



class CrossAttention(nn.Module):
    def __init__(self, c1, c2, num_heads=8, drop_rate=0.1):
        super().__init__()
        c_common = c1
        self.k_proj = nn.Linear(c2, c_common)
        self.v_proj = nn.Linear(c2, c_common)
        self.drop = nn.Dropout(drop_rate)

        self.attn = nn.MultiheadAttention(embed_dim=c_common, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(c_common)

    def forward(self, A, B):

        Q = A
        K = self.k_proj(B)  # [B, L2, C]
        V = self.v_proj(B)

        attn_out, _ = self.attn(Q, K, V)  # [B, L1, C]
        x = self.norm1(self.drop(attn_out) + Q)
        return x  # shape: [B, L1, C_common]



class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out



class Featureformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, history_data: torch.Tensor):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        x = history_data
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1] * self.steps_per_day
        if self.dow_embedding_dim > 0:
            dow = x[..., 2] * 7
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                tod.long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)

        return x





class GraphLLMdualGateSpbise(nn.Module):

    def __init__(self, **model_args):
        super(GraphLLMdualGateSpbise, self).__init__()
        self.task_name = model_args['task_name']
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.d_ff = model_args['d_ff']
        self.top_k = 10
        self.d_llm = model_args['llm_dim']
        self.decoder_layer_num= model_args['d_layer']
        self.encoder_layer_num = model_args['e_layer']
        self.selected_rate =  model_args['prompt_domain']
        self.d_dimff = model_args['d_dimff']
        self.d_nhead = model_args['d_nhead']
        self.e_dimff = model_args['e_dimff']
        self.e_nhead = model_args['e_nhead']
        self.cross_nhead = model_args['cross_nhead']


        if model_args['llm_model'] == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
            self.llama_config.num_hidden_layers = model_args['llm_layers']
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'meta-llama/Meta-Llama-3-8B',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'meta-llama/Meta-Llama-3-8B',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'meta-llama/Meta-Llama-3-8B',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'meta-llama/Meta-Llama-3-8B',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif model_args['llm_model'] == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('gpt2')

            self.gpt2_config.num_hidden_layers = model_args['llm_layers']
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif model_args['llm_model'] == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = model_args['llm_layers']
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.description = model_args['content']
        self.dropout_rate = model_args['dropout']

        self.model_dim = 24*3 + 80
        self.latent_dim = 128
        self.FE = Featureformer(model_args['d_model'])

        self.output_proj =nn.Linear(self.model_dim*model_args['enc_in'], 1*model_args['c_out'])


        self.selected_nums = int(model_args['d_model'] * self.selected_rate)
        config = TokenizerConfig(
            tokenizer_class="MeanScaleUniformBins",
            tokenizer_kwargs=dict(low_limit=-3.0, high_limit=3.0),
            n_tokens=1024,
            n_special_tokens=2,
            pad_token_id=0,
            eos_token_id=1,
            use_eos_token=True,
            model_type="causal",
            context_length=model_args['enc_in']*self.selected_nums
        )
        self.tokenizer_time = config.create_tokenizer()

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_llm, nhead=self.e_nhead,
                                                   dim_feedforward=self.e_dimff, dropout=self.dropout_rate, batch_first=True)
        self.selected_n = NodeSelector()
        self.attn_layers_align = CrossAttention(self.model_dim, self.d_llm, num_heads=self.cross_nhead)
        self.attn_layers_alignbi = CrossAttention(self.d_llm, self.d_llm, num_heads=self.cross_nhead)
        self.attn_layers_self = nn.TransformerEncoder(encoder_layer, num_layers=self.encoder_layer_num)
        self.attn_layers_self2 = nn.TransformerEncoder(encoder_layer, num_layers=self.encoder_layer_num)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.model_dim, nhead=self.d_nhead,
                                                   dim_feedforward=self.d_dimff, dropout=self.dropout_rate, batch_first=True)
        self.out_fusion = nn.TransformerDecoder(decoder_layer, num_layers=self.decoder_layer_num)
        self.seblock = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.model_dim // 4, self.model_dim),
            nn.Sigmoid()
        )




    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
   
        B, T, N, C = history_data.shape
        x_enc = history_data[:, :, :, 0]
        x_feature = self.FE(history_data)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':

            dec_out = self.forecast(x_enc, x_feature)
            return dec_out
        return None

    def forecast(self, x_enc, x_feature):
        B, T, N = x_enc.size()
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        avg_values = torch.mean(x_enc, dim=1)
        medians = torch.median(x_enc, dim=1).values

        min_values_spa = torch.min(x_enc, dim=2)[0]
        max_values_spa = torch.max(x_enc, dim=2)[0]
        medians_spa = torch.median(x_enc, dim=2).values
        avg_values_spa = torch.mean(x_enc, dim=2)
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B, N * T, 1)
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
        # print(min_values)
        # print(avg_values)
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            avg_values_str = str(avg_values[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            min_values_str_spa = str(min_values_spa[b].tolist()[0])
            max_values_str_spa = str(max_values_spa[b].tolist()[0])
            median_values_str_spa = str(medians_spa[b].tolist()[0])
            avg_values_spa_str = str(avg_values_spa[b].tolist()[0])
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps on {str(N)} road sensors given the previous {str(self.seq_len)} steps information on {str(N)} road sensors; "
                "Input statistics: "
                f"temporal min value {min_values_str}, "
                f"spatial min value {min_values_str_spa}, "
                f"temporal max value {max_values_str}, "
                f"spatial max value {max_values_str_spa}, "
                f"temporal median value {median_values_str}, "
                f"spatial median value {median_values_str_spa}, "
                f"temporal average value {avg_values_str}, "
                f"spatial average value {avg_values_spa_str}, "
                f"the spatiotemporal trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 10 spatiotemporal lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)


        x_enc = x_enc.reshape(B, N, T).contiguous()
        scores = self.selected_n(x_enc)
        topkids, _ = gumbel_softmax_topk(scores, self.selected_nums)
        x_enc = extract_subgraph(x_enc, topkids)
        x_enc = x_enc.reshape(B, self.selected_nums * T)
        input_ids_x_enc, _, _ = self.tokenizer_time.input_transform(x_enc)
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        st_llm_embeddings = self.llm_model.get_input_embeddings()(input_ids_x_enc)
        # print(st_llm_embeddings.shape)
        x_feature = x_feature.permute(0, 2, 1, 3).contiguous()
        x_res = torch.reshape(x_feature, (B, N*T, self.model_dim))
        llm_prompt_out = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
        llm_st_out = self.llm_model(inputs_embeds=st_llm_embeddings).last_hidden_state
        llm_prompt_out = self.attn_layers_self(llm_prompt_out)
        llm_st_out = self.attn_layers_self2(llm_st_out)
        llm_prompt_out = self.attn_layers_alignbi(llm_prompt_out, llm_st_out)
        llm_st_out = self.attn_layers_alignbi(llm_st_out, llm_prompt_out)
        llm_align_out_p = self.attn_layers_align(x_res, llm_prompt_out)
        llm_align_out_st =  self.attn_layers_align(x_res, llm_st_out)
        llm_fusion = self.out_fusion(self.seblock(llm_align_out_p) *x_res + self.seblock(llm_align_out_st) * x_res,
                                     self.seblock(llm_align_out_p) *x_res + self.seblock(llm_align_out_st) * x_res)
        llm_fusion = llm_fusion.reshape(B, N, T, self.model_dim)
        llm_fusion = llm_fusion.reshape(B, N, T*self.model_dim)
        llm_fusion = self.output_proj(llm_fusion).reshape(B, N, T, 1)
        final_dec_out = llm_fusion.transpose(1, 2) # (B, L_out, N,  1)
        return final_dec_out



    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


