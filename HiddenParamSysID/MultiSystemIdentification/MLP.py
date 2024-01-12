from typing import List, Tuple, Any

from MultiSystemIdentification.Predictor import Predictor
import torch

class MLP(Predictor):
    def __init__(self, input_size, output_size, device="cuda:0", hidden_size=1000):
        super().__init__(input_size, output_size, device=device)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self,
              example_xs: torch.tensor,
              example_ys: torch.tensor,
              xs: torch.tensor,
              ys: torch.tensor) -> Tuple[float, Any]:
        assert xs.shape[-1] == self.input_size, f"Input size of model '{self.input_size}' does not match input size of data '{xs.shape[1]}'"
        assert ys.shape[-1] == self.output_size, f"Output size of model '{self.output_size}' does not match output size of data '{ys.shape[1]}'"

        # backprop
        self.optimizer.zero_grad()

        # due to the size of the data, we need to do gradient accumulation
        max_batch = 4 # max number of functions per gradient calculation
        number_batches = int(example_xs.shape[0] / max_batch)
        assert example_xs.shape[0] % max_batch == 0, f"example_xs.shape[0] ({example_xs.shape[0]}) must be divisible by max_batch ({max_batch})"

        total_loss = 0
        for batch_number in range(number_batches):
            xs_batch = xs[batch_number * max_batch : (batch_number + 1) * max_batch]
            ys_batch = ys[batch_number * max_batch : (batch_number + 1) * max_batch]

            # compute loss
            y_hat = self.model(xs_batch)
            assert y_hat.shape == ys_batch.shape, f"y_hat is wrong shape, got {y_hat.shape}, expected {ys.shape}"
            loss = torch.nn.MSELoss()(y_hat, ys_batch)

            loss.backward() # accumulate grads
            total_loss += loss.item()

        # update params
        self.optimizer.step()
        return total_loss/number_batches, None

    def test(self,
             example_xs: torch.tensor,
             example_ys: torch.tensor,
             xs: torch.tensor,
             ys: torch.tensor) -> float:
        assert xs.shape[-1] == self.input_size, f"Input size of model '{self.input_size}' does not match input size of data '{xs.shape[1]}'"
        assert ys.shape[-1] == self.output_size, f"Output size of model '{self.output_size}' does not match output size of data '{ys.shape[1]}'"

        with torch.no_grad():
            # due to the size of the data, we need to do gradient accumulation
            max_batch = 4  # max number of functions per gradient calculation
            number_batches = int(example_xs.shape[0] / max_batch)
            assert example_xs.shape[
                       0] % max_batch == 0, f"example_xs.shape[0] ({example_xs.shape[0]}) must be divisible by max_batch ({max_batch})"

            total_loss = 0
            for batch_number in range(number_batches):
                xs_batch = xs[batch_number * max_batch: (batch_number + 1) * max_batch]
                ys_batch = ys[batch_number * max_batch: (batch_number + 1) * max_batch]

                # compute loss
                y_hat = self.model(xs_batch)
                assert y_hat.shape == ys_batch.shape, f"y_hat is wrong shape, got {y_hat.shape}, expected {ys.shape}"
                loss = torch.nn.MSELoss()(y_hat, ys_batch)

                total_loss += loss.item()
            return total_loss / number_batches

    def forward_testing(self,
                example_xs: torch.tensor, # F x B1 x SA size
                example_ys: torch.tensor, # F x B1 x S size
                xs: torch.tensor # F X B2 x SA size
                ) -> torch.Tensor:

        # compute loss
        y_hat = self.model(xs)
        return y_hat