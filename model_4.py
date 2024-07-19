from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


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
    def __init__(self, vocabulary_size):
        super().__init__()

        self.vocabulary_size = vocabulary_size

    def freeze(self):
        """Sets the requires_grad parameter to False for some model parameters."""
        raise NotImplementedError

    def forward(self, encoded_image, caption_indices, *args):
        """Forward method.

        :param encoded_image: torch.tensor of the shape [batch_size, *] or None
        :param caption_indices: torch.tensor of the shape [batch_size, sequence_length] or None
        :param args: e.g., hidden state

        :return: output dict at least with 'logits' and 'indices' keys,
            where: logits is the torch.tensor of the shape [batch_size, vocabulary_size, sequence_length]
                   indices is the torch.tensor of the shape [batch_size, sequence_length]
        """
        raise NotImplementedError

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        """Generates caption indices like torch.tensor([1, 23, 5, 8, 2]).

        :param encoded_image: torch.tensor of the shape [1, *]
        :param sos_token_index: index of the "start of sequence" token (int)
        :param eos_token_index: index of the "end of sequence" token (int)
        :param max_length: maximum caption length (int)

        :return: caption indices (list of the length <= max_length)
        """
        raise NotImplementedError
