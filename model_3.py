from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator
import torch.nn as nn
import torchvision.models
import torch
from typing import Tuple
import torch.nn.functional as F
from einops import rearrange, pack

class Model(BaseModel):
    """Base class for all models."""
    def __init__(self, vocabulary, embedding_dim, num_layers):
        super().__init__(vocabulary=vocabulary)

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.image_encoder = ImageEncoder(embedding_dim=self.embedding_dim)
        self.caption_generator = CaptionGenerator(vocabulary_size=len(self.vocabulary),
                                                  embedding_dim=self.embedding_dim,
                                                  hidden_dim=self.embedding_dim,
                                                  num_layers=self.num_layers)



class ImageEncoder(BaseImageEncoder):
    def __init__(self, embedding_dim):
        super().__init__()

        

        # Load the DINOv2 model
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.embed_dim = self.dino.embed_dim
        # Adjust the output projection layer to match the desired embedding dimension
        self.projection = nn.Linear(self.embed_dim, embedding_dim)
    def freeze(self):
        """Sets the requires_grad parameter to False for some model parameters."""
        for param in self.dino.parameters():
            param.requires_grad = False
        #raise NotImplementedError

    def forward(self, image: torch.Tensor, scale: int = 1) -> torch.Tensor:
        """Forward method.

        :param image: torch.tensor of the shape [batch_size, channels, height, width]

        :return: encoded image (torch.tensor) of the shape [batch_size, *]
        """
        # Resize the input image to a multiple of 224*224
        x = F.interpolate(image, size=(scale * 224, scale * 224), mode="bilinear", align_corners=False)
        
        # Get the intermediate layers from DINOv2
        out = self.dino.get_intermediate_layers(x, n=1, reshape=True, return_class_token=True)[0]

        # Use the class token as the image representation
        image_representation = out[1]

        # Project the image representation to the desired embedding dimension
        image_embedding = self.projection(image_representation)

        return image_embedding


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Sequential(torch.nn.Embedding(num_embeddings=self.vocabulary_size,
                                                                embedding_dim=self.embedding_dim),
                                             torch.nn.Dropout(0.5))

        self.GRU = torch.nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                                num_layers=self.num_layers,
                                batch_first=True)
        self.to_logits = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.vocabulary_size)

    def freeze(self):
        """Sets the requires_grad parameter to False for some model parameters."""
        for param in self.GRU.parameters():
            param.requires_grad = False
          
    def _get_embeddings(self, encoded_image=None, caption_indices=None):
        if caption_indices is None:
            embeddings = rearrange(encoded_image, 'batch embedding_dim -> batch 1 embedding_dim')
        else:
            embeddings = self.embedding(caption_indices)
            if encoded_image is not None:
                embeddings, _ = pack([encoded_image, embeddings], 'batch * embedding_dim')

        return embeddings
        

 
    def forward(self, encoded_image, caption_indices, hidden_state=None):
        """Forward method.

        :param encoded_image: torch.tensor of the shape [batch_size, *] or None
        :param caption_indices: torch.tensor of the shape [batch_size, sequence_length] or None
        :param args: e.g., hidden state

        :return: output dict at least with 'logits' and 'indices' keys,
            where: logits is the torch.tensor of the shape [batch_size, vocabulary_size, sequence_length]
                   indices is the torch.tensor of the shape [batch_size, sequence_length]
        """
        if encoded_image is not None and caption_indices is not None:
            caption_indices = caption_indices[:, 1:]  # the encoded image will be used instead of the <SOS> token

        embeddings = self._get_embeddings(encoded_image=encoded_image, caption_indices=caption_indices)

        output, hidden_state = self.GRU(input=embeddings, hx=hidden_state)
        logits = self.to_logits(output)
        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        return {'logits': logits, 'indices': logits.argmax(dim=-2), 'hidden_state': hidden_state}


    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        """Generates caption indices like torch.tensor([1, 23, 5, 8, 2]).

        :param encoded_image: torch.tensor of the shape [1, *]
        :param sos_token_index: index of the "start of sequence" token (int)
        :param eos_token_index: index of the "end of sequence" token (int)
        :param max_length: maximum caption length (int)

        :return: caption indices (list of the length <= max_length)
        """
        caption_indices = []

        output = self.forward(encoded_image, caption_indices=None, hidden_state=None)
        for _ in range(max_length):
            predicted_index = output['indices']

            caption_indices.append(predicted_index.item())
            if predicted_index.item() == eos_token_index:
                break

            output = self.forward(encoded_image=None,
                                  caption_indices=predicted_index,
                                  hidden_state=output['hidden_state'])

        return caption_indices


