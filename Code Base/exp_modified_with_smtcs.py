# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================
# exp.py
# Description: Implementation of EXP algorithm
# ============================================
import numpy as np
import torch
import scipy
from math import log
from transformers import AutoModel, AutoTokenizer
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from visualize.data_for_visualization import DataForVisualization
import logging

# Configure logging to write to a file
logging.basicConfig(filename='watermark_detection.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
        """Seed the random number generator with the last `prefix_length` tokens or embedding, with scaling."""
        time_result = 1
        scale_factor = 1.0  # Default scaling factor

        if input_ids.dim() == 2:
            input_ids = input_ids[0]  # Select the first item from the batch if batch size > 1

        # If input contains floats, calculate scaling factor
        if input_ids.dtype != torch.long:
            abs_input_ids = input_ids.abs()  # Get absolute values of the tensor
            min_val = abs_input_ids.min().item()
            if min_val <= 0:  # Ensure non-zero positive values
                raise ValueError("Tensor contains non-positive values, log cannot be computed.")

            # Calculate scaling factor based on the minimum value in the input tensor
            n = -log(min_val) + 1
            scale_factor = 10 ** n
            input_ids = input_ids * scale_factor  # Apply scaling to the tensor

        # Calculate scalar value based on the last `prefix_length` tokens
        prefix_length = min(self.config.prefix_length, len(input_ids))
        for i in range(prefix_length):
            time_result *= input_ids[-1 - i].item()

        # Use a combination of the hash key and the scalar value to seed the RNG
        prev_token = int(time_result % self.config.vocab_size)
        seed = self.config.hash_key * prev_token
        self.rng.manual_seed(seed)

        return scale_factor  # Return the scaling factor



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
    """Top-level class for the EXP algorithm with cross-lingual robustness."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs):
        """
        Initialize the EXP algorithm with cross-lingual robustness.

        Parameters:
            algorithm_config (str): Path to the algorithm configuration file.
            transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        super().__init__(algorithm_config, transformers_config)
        self.config = EXPConfig(algorithm_config, transformers_config)
        self.utils = EXPUtils(self.config)

        # Load a multilingual embedding model for cross-lingual robustness
        self.multilingual_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.multilingual_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.multilingual_model.to(self.config.device)

    def get_semantic_embedding(self, text: str):
        """Get the semantic embedding of the text using a multilingual model."""
        inputs = self.multilingual_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.config.device)
        with torch.no_grad():
            outputs = self.multilingual_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Mean pooling for sentence embedding

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> tuple:
        """Generate watermarked text using the EXP algorithm with cross-lingual robustness and scaling."""
        # Get semantic embedding of the prompt
        semantic_embedding = self.get_semantic_embedding(prompt)

        # Seed RNG with the semantic embedding and get scaling factor
        scale_factor = self.utils.seed_rng(semantic_embedding)

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)

        # Initialize
        inputs = encoded_prompt
        attn = torch.ones_like(encoded_prompt)
        past = None

        # Generate tokens
        for i in range(self.config.sequence_length):
            with torch.no_grad():
                if past:
                    output = self.config.generation_model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
                else:
                    output = self.config.generation_model(inputs)

            # Get probabilities
            probs = torch.nn.functional.softmax(output.logits[:, -1, :self.config.vocab_size], dim=-1).cpu()

            # Generate r1, r2,..., rk
            self.utils.seed_rng(inputs[0])  # Re-seed the RNG with the current input
            random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.rng)

            # Sample token to add watermark
            token = self.utils.exp_sampling(probs, random_numbers).to(self.config.device)

            # Update inputs
            inputs = torch.cat([inputs, token], dim=-1)

            # Update past
            past = output.past_key_values

            # Update attention mask
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        watermarked_tokens = inputs[0].detach().cpu()
        watermarked_text = self.config.generation_tokenizer.decode(watermarked_tokens, skip_special_tokens=True)
        return watermarked_text


    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> dict:
        """Detect watermark in the text with cross-lingual robustness and scaling."""
        # Get semantic embedding of the text
        semantic_embedding = self.get_semantic_embedding(text)

        # Seed RNG with the semantic embedding and get scaling factor
        scale_factor = self.utils.seed_rng(semantic_embedding)

        # Encode the text into tokens using the configured tokenizer
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(self.config.device)

        # Calculate the number of tokens to score, excluding the prefix
        num_scored = encoded_text.size(1) - self.config.prefix_length
        total_score = 0

        for i in range(self.config.prefix_length, encoded_text.size(1)):
            # Seed RNG with the prefix of the encoded text
            self.utils.seed_rng(encoded_text[:, :i])

            # Generate random numbers for each token in the vocabulary
            random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.rng)

            # Calculate score for the current token
            token = int(encoded_text[0, i].item())  # Extract token ID from tensor
            r = random_numbers[token]
            total_score += log(1 / r.item())

        # Normalize the score by the number of tokens scored
        normalized_score = total_score / num_scored if num_scored > 0 else 0

        # Compare normalized score to the threshold to determine if it's watermarked
        is_watermarked = normalized_score > self.config.threshold

        return {"watermark_detected": is_watermarked, "score": normalized_score} if return_dict else is_watermarked


         
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        """Get data for visualization."""

        # Encode the text into tokens using the configured tokenizer
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).numpy()[0]

        # Initialize the list of values with None for the prefix length
        highlight_values = [None] * self.config.prefix_length

        # Calculate the value for each token beyond the prefix
        for i in range(self.config.prefix_length, len(encoded_text)):
            # Seed the random number generator using the prefix of the encoded text
            self.utils.seed_rng(encoded_text[:i])
            random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.rng)
            r = random_numbers[encoded_text[i]]
            v = log(1 / (1 - r))
            v = self.utils._value_transformation(v)
            highlight_values.append(v)

        # Decode each token id to its corresponding string token
        decoded_tokens = [self.config.generation_tokenizer.decode([token_id]) for token_id in encoded_text]
        
        return DataForVisualization(decoded_tokens, highlight_values)
        
    