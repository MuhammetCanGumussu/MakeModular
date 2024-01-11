"""
Contains Pytorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn

class TinyVGG(nn.Module):
    """
    Creates the TinyVGG architecture

    Replicates the TinyVGG architecture from the CNN explainer website in
    Pytorch. See the original architecutre here: https://poloclub.github.io/cnn-explainer/

    Args(consturcture):
        input_shape: An integer indicating number of input channels.
        hidden_units: An integer indicating number of hidden units between layers
        output_shape: An integer indicating number of output units.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                     out_channels=hidden_units,
                     kernel_size=3,
                     stride=1,
                     padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classfier = nn.Sequential(
            nn.Flatten(),
            # in_features shape derived from error message by trying model with 
            # trial-garbage-experimental tensor
            nn.Linear(in_features=hidden_units*13*13,
                     out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        # for leveraging operator fusion optimization
        return self.classfier(self.conv_block_2(self.conv_block_1(x))) 
        #x = self.conv_block_1(x)
        #x = self.conv_block_2(x)
        #x = self.classfier(x)
        #return x
class LargeVGG(nn.Module):
    """
    Creates the LargeVGG architecture

    Replicates the TinyVGG architecture from the CNN explainer website in
    Pytorch. See the original architecutre here: https://poloclub.github.io/cnn-explainer/

    Args(consturcture):
        input_shape: An integer indicating number of input channels.
        hidden_units: An integer indicating number of hidden units between layers
        output_shape: An integer indicating number of output units.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                     out_channels=hidden_units,
                     kernel_size=3,
                     stride=1,
                     padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units*2,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units*2,
                     out_channels=hidden_units*2,
                     kernel_size=3,
                     stride=1,
                     padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*2,
                      out_channels=hidden_units*4,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units*4,
                     out_channels=hidden_units*4,
                     kernel_size=3,
                     stride=1,
                     padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(hidden_units*4, hidden_units*8, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_units*8, hidden_units*8, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classfier = nn.Sequential(
            nn.Flatten(),
            # in_features shape derived from error message by trying model with 
            # trial-garbage-experimental tensor
            nn.Linear(in_features=1280,
                     out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        # for leveraging operator fusion optimization
        return self.classfier(self.conv_block_4(self.conv_block_3(self.conv_block_2(self.conv_block_1(x)))))
  
