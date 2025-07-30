import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


@dataclass
class TokenizerConfig:
   

    tokenizer_class: str
    tokenizer_kwargs: Dict[str, Any]
    n_tokens: int
    n_special_tokens: int
    pad_token_id: int
    eos_token_id: int
    use_eos_token: bool
    model_type: Literal["causal", "seq2seq"]
    context_length: int

    def __post_init__(self):
        assert (
                self.pad_token_id < self.n_special_tokens
                and self.eos_token_id < self.n_special_tokens
        ), f"Special token id's must be smaller than {self.n_special_tokens=}"

    def create_tokenizer(self) -> "ChronosTokenizer":
        if self.tokenizer_class == "MeanScaleUniformBins":
            return MeanScaleUniformBins(**self.tokenizer_kwargs, config=self)
        raise ValueError


class ChronosTokenizer:
 

    def input_transform(
            self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
      
        raise NotImplementedError()

    def output_transform(
            self, samples: torch.Tensor, tokenizer_state: Any
    ) -> torch.Tensor:
      
        raise NotImplementedError()


class MeanScaleUniformBins(ChronosTokenizer):
    def __init__(
            self, low_limit: float, high_limit: float, config: TokenizerConfig
    ) -> None:
        self.config = config
        self.centers = torch.linspace(
            low_limit,
            high_limit,
            config.n_tokens - config.n_special_tokens - 1,
        ).cuda()
        self.boundaries = torch.concat(
            (
                torch.tensor([-1e20], device=self.centers.device),
                (self.centers[1:] + self.centers[:-1]) / 2,
                torch.tensor([1e20], device=self.centers.device),
            )
        )

    def input_transform(
            self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, length = context.shape

        if length > self.config.context_length:
            context = context[..., -self.config.context_length:]

        attention_mask = ~torch.isnan(context)
        scale = torch.nansum(
            torch.abs(context) * attention_mask, dim=-1
        ) / torch.nansum(attention_mask, dim=-1)
        scale[~(scale > 0)] = 1.0
        scaled_context = context / scale.unsqueeze(dim=-1)
        # print(self.boundaries)
        token_ids = (
                torch.bucketize(
                    input=scaled_context,
                    boundaries=self.boundaries,
                    right=True,
                )
                + self.config.n_special_tokens
        )
        token_ids[~attention_mask] = self.config.pad_token_id

        if self.config.use_eos_token:
            eos_tokens = torch.full(
                (batch_size, 1), fill_value=self.config.eos_token_id
            ).cuda()
            token_ids = torch.concat((token_ids, eos_tokens), dim=1).cuda()
            eos_mask = torch.full((batch_size, 1), fill_value=True).cuda()
            attention_mask = torch.concat((attention_mask, eos_mask), dim=1).cuda()

        return token_ids, attention_mask, scale

    def output_transform(
            self, samples: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:

        scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
        indices = torch.clamp(
            samples - self.config.n_special_tokens,
            min=0,
            max=len(self.centers) - 1,
        ).long()
        self.centers = self.centers.to(samples.device)
        return self.centers[indices] * scale_unsqueezed
