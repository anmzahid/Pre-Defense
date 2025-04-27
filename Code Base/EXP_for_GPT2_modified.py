import torch
import scipy
from math import log
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from visualize.data_for_visualization import DataForVisualization
from transformers import AutoTokenizer, AutoModelForCausalLM


class EXPConfig:
    """Config class for EXP algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/EXP.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'EXP':
            raise AlgorithmNameMismatchError('EXP', config_dict['algorithm_name'])

        self.prefix_length = config_dict['prefix_length']
        self.hash_key = config_dict['hash_key']
        self.threshold = config_dict['threshold']
        self.sequence_length = config_dict['sequence_length']
        self.top_k = config_dict['top_k']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class EXPUtils:
    """Utility class for EXP algorithm, contains helper functions."""

    def __init__(self, config: EXPConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP utility class.

            Parameters:
                config (EXPConfig): Configuration for the EXP algorithm.
        """
        self.config = config
        self.rng = torch.Generator()

    def seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the random number generator with the last `prefix_length` tokens of the input."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.rng.manual_seed(self.config.hash_key * prev_token)
        return
    
    def exp_sampling(self, probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Sample a token from the vocabulary using the exponential sampling method."""
        
        # If top_k is not specified, use argmax
        if self.config.top_k <= 0:
            return torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)
        
        # Ensure top_k is not greater than the vocabulary size
        top_k = min(self.config.top_k, probs.size(-1))
    
        # Get the top_k probabilities and their indices
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
    
        # Perform exponential sampling on the top_k probabilities
        sampled_indices = torch.argmax(u.gather(-1, top_indices) ** (1 / top_probs), dim=-1)
    
        # Map back the sampled indices to the original vocabulary indices
        return top_indices.gather(-1, sampled_indices.unsqueeze(-1))
    
    def _value_transformation(self, value):
        """Transform the value to a range between 0 and 1."""
        return value / (value + 1)
    

class EXP(BaseWatermark):
    """Top-level class for the EXP algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = EXPConfig(algorithm_config, transformers_config)
        self.utils = EXPUtils(self.config)


def generate_watermarked_text(prompt: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, device: torch.device, max_length: int = 175, **kwargs):
    """Generate watermarked text using EXP algorithm."""
    
    # Encode prompt to tensor format
    encoded_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Initialize attention mask and past_key_values
    attn = torch.ones_like(encoded_prompt).to(device)
    past = None

    # Generate tokens (looping through to generate new tokens step by step)
    for i in range(max_length):
        with torch.no_grad():
            if past:
                output = model(encoded_prompt[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(encoded_prompt)
        
        # Get probabilities for the next token
        logits = output.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu()
        
        # Sample a token for watermarking (Here we can introduce randomness for watermarking)
        token = torch.multinomial(probs, 1).to(device)
        
        # Append the sampled token to the inputs
        encoded_prompt = torch.cat([encoded_prompt, token], dim=-1)

        # Update past key values and attention mask
        past = output.past_key_values
        attn = torch.cat([attn, torch.ones((attn.shape[0], 1)).to(device)], dim=-1)
    
    # Convert the final tokens to text (after watermarking)
    watermarked_tokens = encoded_prompt[0].detach().cpu()
    watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)
    
    return watermarked_text  


def detect_watermark(text: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, config: EXPConfig, utils: EXPUtils, return_dict: bool = True, *args, **kwargs) -> dict:
    """Detect watermark in the text."""

    # Encode the text into tokens using the configured tokenizer
    encoded_text = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(config.device).cpu().numpy()[0]

    # Calculate the number of tokens to score, excluding the prefix
    num_scored = len(encoded_text) - config.prefix_length
    total_score = 0

    for i in range(config.prefix_length, len(encoded_text)):
        # Seed RNG with the prefix of the encoded text
        utils.seed_rng(encoded_text[:i])

        # Generate random numbers for each token in the vocabulary
        random_numbers = torch.rand(config.vocab_size, generator=utils.rng).to(config.device)

        # Calculate score for the current token
        r = random_numbers[encoded_text[i]]
        total_score += log(1 / (1 - r))

    # Calculate p_value
    p_value = scipy.stats.gamma.sf(total_score, num_scored, loc=0, scale=1)

    # Determine if the computed score exceeds the threshold for watermarking
    is_watermarked = p_value < config.threshold

    # Return results based on the `return_dict` flag
    if return_dict:
        return {"is_watermarked": is_watermarked, "score": p_value}
    else:
        return (is_watermarked, p_value)


def get_data_for_visualization(text: str, tokenizer: AutoTokenizer, config: EXPConfig, utils: EXPUtils, *args, **kwargs) -> DataForVisualization:
    """Get data for visualization."""

    # Encode the text into tokens using the configured tokenizer
    encoded_text = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(config.device).cpu().numpy()[0]

    # Initialize the list of values with None for the prefix length
    highlight_values = [None] * config.prefix_length

    # Calculate the value for each token beyond the prefix
    for i in range(config.prefix_length, len(encoded_text)):
        # Seed the random number generator using the prefix of the encoded text
        utils.seed_rng(encoded_text[:i])
        random_numbers = torch.rand(config.vocab_size, generator=utils.rng)
        r = random_numbers[encoded_text[i]]
        v = log(1 / (1 - r))
        v = utils._value_transformation(v)
        highlight_values.append(v)

    # Decode each token id to its corresponding string token
    decoded_tokens = [tokenizer.decode([token_id]) for token_id in encoded_text]
    
    return DataForVisualization(decoded_tokens, highlight_values)
