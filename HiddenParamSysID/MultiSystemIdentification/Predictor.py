import os
from typing import List, Tuple, Any
import torch

# the virtual class for predictor models.
class Predictor:
    def __init__(self, input_size, output_size, device):
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

    # this function should take one gradient descent step using the data provided. The example data is data
    # to be used to guide the prediction, IE it is information on the function to predict implicitly. The
    # xs is the input we want to predict an output for. The ys is the output we want to
    # predict. Returns the train MSE of the prediction.
    def train(self,
              example_xs:torch.tensor,
              example_ys:torch.tensor,
              xs:torch.tensor,
              ys:torch.tensor) -> Tuple[float, Any]:
        pass


    # Same as above, but this should return the MSE of the test prediction without doing anything else.
    def test(self,
             example_xs: torch.tensor,
             example_ys: torch.tensor,
             xs: torch.tensor,
             ys: torch.tensor) -> float:
        pass

    # returns the number of parameters of the internal NN. It must be named self.model.
    def num_params(self) -> int:
        assert self.model
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))

    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pt")))

    def forward_testing(self,
                example_xs: torch.tensor, # F x B1 x SA size
                example_ys: torch.tensor, # F x B1 x S size
                xs: torch.tensor # F X B2 x SA size
                ) -> torch.Tensor:
        pass
