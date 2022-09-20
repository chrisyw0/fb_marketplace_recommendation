import torch

from typing import List, Tuple, Any, Dict
from fbRecommendation.dl.base.utils.image_text_util import ImageTextUtil

from torch.nn.utils.rnn import pad_sequence


class PTImageTextUtil(ImageTextUtil):
    """
    This class is a subclass ImageTextUtil, provide utils functions for tensorflow model
    """

    @staticmethod
    def prepare_token_with_padding(
            X: Tuple[List[int], List[int], List[int]], embedding: str
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Add padding to the tokens. This is developed for non-transformer based data only. To add padding for transformer based
        model, use batch_encode_text instead.

        Args:
            X: A tuple containing training, validation and testing (in List[int] format), to be padded with trailing
               zero
            embedding: The embedding method. Currently, support Word2Vec only

        Returns:
            A tuple of training, validation and testing dataset

        """
        X_train, X_val, X_test = X

        if embedding == "Word2Vec":
            # add padding
            text_with_padding = [torch.IntTensor(x) for x in X_train]
            text_with_padding.extend([torch.IntTensor(x) for x in X_val])
            text_with_padding.extend([torch.IntTensor(x) for x in X_test])

            text_with_padding = pad_sequence(text_with_padding, True, 0)

            train_end_idx = len(X_train)
            val_end_idx = len(X_train) + len(X_val)

            X_train = text_with_padding[:train_end_idx]
            X_val = text_with_padding[train_end_idx:val_end_idx]
            X_test = text_with_padding[val_end_idx:]

            return X_train, X_val, X_test

    @staticmethod
    def batch_encode_text(
            X: Tuple[List[str], List[str], List[str]], tokenizer: Any
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Encode the text into token ids with padding. This is developed for transformer based data only. To add padding
        for non-transformer based model, use prepare_token_with_padding instead.
        Args:
            X: A tuple containing training, validation and testing (in List[int] format), to be padded with trailing
               zero
            tokenizer: Tokeniser that matches the embedding model

        Returns:
            Tuple[Dict, Dict, Dict]:
                A tuple of dictionary containing the encoded and padded data of training, validation and testing data.

        """

        X_train, X_val, X_test = X

        train_end_idx = len(X_train)
        val_end_idx = len(X_train) + len(X_val)

        text = X_train
        text.extend(X_val)
        text.extend(X_test)

        encoded_text = tokenizer.batch_encode_plus(
            text,
            padding="max_length",
            truncation=True
        )

        encoded_text = {key: torch.LongTensor(value) for key, value in encoded_text.items()}

        X_train = {key: value[:train_end_idx] for key, value in encoded_text.items()}
        X_val = {key: value[train_end_idx:val_end_idx] for key, value in encoded_text.items()}
        X_test = {key: value[val_end_idx:] for key, value in encoded_text.items()}

        return X_train, X_val, X_test
