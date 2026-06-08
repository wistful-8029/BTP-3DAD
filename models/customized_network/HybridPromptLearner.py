import torch.nn as nn
import torch
from typing import Union, List
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from packaging import version

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[
    torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class HybridPromptLearner(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)

                                                     
        self.n_ctx = design_details.get("Prompt_length", 1)

        dtype = clip_model.transformer.get_cast_dtype()
        ctx_dim = clip_model.ln_final.weight.shape[0]

                                         
        self.ctx = nn.Parameter(torch.empty(self.n_ctx, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx, std=0.02)

                                 
                                    
                                                
           
                                     
                                                   
           

                                 
        self.state_normal_list = [
            "normal {}."
        ]
        self.state_anomaly_list = [
            "defective {}."
        ]

                                                     
        self.classnames = [name.replace("_", " ") for name in classnames]

                                  
        def build_prompts(templates, classnames):
            texts = [template.format("") for template in templates for name in classnames]
            tokenized = torch.cat([tokenize(t) for t in texts])
            with torch.no_grad():
                embeddings = clip_model.token_embedding(tokenized).type(dtype)             
            return tokenized, embeddings

        self.tokenized_prompts_pos, embedding_pos = build_prompts(self.state_normal_list, self.classnames)
        self.tokenized_prompts_neg, embedding_neg = build_prompts(self.state_anomaly_list, self.classnames)

                       
        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :])         
        self.register_buffer("token_suffix_pos", embedding_pos[:, 1:, :])        
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, 1:, :])

    def forward(self):
                                          
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

                                                                              
        prompts_pos = torch.cat([self.token_prefix_pos, ctx, self.token_suffix_pos], dim=1)
        prompts_neg = torch.cat([self.token_prefix_neg, ctx, self.token_suffix_neg], dim=1)

        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)
        tokenized = torch.cat([self.tokenized_prompts_pos, self.tokenized_prompts_neg], dim=0)

        return prompts, tokenized
