from typing import List, Tuple

from MultiSystemIdentification.Predictor import Predictor
import torch


'''
This is the FE_Dif predictor. It predicts the difference between the average function and the current function,
instead of the function itself, which requires less data to make an accurate prediction.
It learns both the average function as a MLP and the difference as a FE. 
With 200 datapoints, it beats the performance of a Transformer with the same amount of data. 
'''
class FE_Dif(Predictor):
    def __init__(self, input_size, output_size, embed_size=100, device="cuda:0", hidden_size=700):
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
            torch.nn.Linear(hidden_size, output_size * embed_size),
        ).to(device)
        self.average_function = torch.nn.Sequential(
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
        self.optimizer_average_function = torch.optim.Adam(self.average_function.parameters(), lr=1e-3)
        self.embed_size = embed_size

    def train(self,
              example_xs: torch.tensor,
              example_ys: torch.tensor,
              xs: torch.tensor,
              ys: torch.tensor) -> Tuple[float, float]:
        assert xs.shape[-1] == self.input_size, f"Input size of model '{self.input_size}' does not match input size of data '{xs.shape[1]}'"
        assert ys.shape[-1] == self.output_size, f"Output size of model '{self.output_size}' does not match output size of data '{ys.shape[1]}'"
        
        # first update the average function
        self.update_average_function(xs, ys)

        # now do the normal one
        # backprop
        self.optimizer.zero_grad()

        # due to the size of the data, we need to do gradient accumulation
        max_batch = 2  # max number of functions per gradient calculation
        number_batches = int(example_xs.shape[0] / max_batch)
        assert example_xs.shape[0] % max_batch == 0, f"example_xs.shape[0] ({example_xs.shape[0]}) must be divisible by max_batch ({max_batch})"

        total_loss = 0
        for batch_number in range(number_batches):
            # get batches
            xs_batch = xs[batch_number * max_batch: (batch_number + 1) * max_batch]
            ys_batch = ys[batch_number * max_batch: (batch_number + 1) * max_batch]
            example_xs_batch = example_xs[batch_number * max_batch: (batch_number + 1) * max_batch]
            example_ys_batch = example_ys[batch_number * max_batch: (batch_number + 1) * max_batch]

            # get encodings from example data
            individual_encoding = self.model(example_xs_batch)
            assert individual_encoding.shape == (example_xs_batch.shape[0], example_xs_batch.shape[1], self.output_size * self.embed_size)
            individual_encoding = individual_encoding.reshape(individual_encoding.shape[0], individual_encoding.shape[1], self.output_size, -1)
            assert individual_encoding.shape == (example_xs_batch.shape[0], example_xs_batch.shape[1], self.output_size, self.embed_size)

            with torch.no_grad():
                example_ys_average = self.average_function(example_xs_batch)
            encodings = torch.mean(individual_encoding * (example_ys_batch - example_ys_average).unsqueeze(-1), dim=1)
            assert encodings.shape == (example_xs_batch.shape[0], self.output_size, self.embed_size)

            # use encodings to make prediction
            train_individual_encodings = self.model(xs_batch)
            assert train_individual_encodings.shape == (xs_batch.shape[0], xs_batch.shape[1], self.output_size * self.embed_size)
            train_individual_encodings = train_individual_encodings.reshape(train_individual_encodings.shape[0], train_individual_encodings.shape[1], self.output_size, -1)
            assert train_individual_encodings.shape == (xs_batch.shape[0], xs_batch.shape[1], self.output_size, self.embed_size)
            y_hat = torch.sum(train_individual_encodings * encodings.unsqueeze(1), dim=-1)
            assert y_hat.shape == ys_batch.shape, f"y_hat is wrong shape, got {y_hat.shape}, expected {ys.shape}"
            with torch.no_grad():
                y_hat_average = self.average_function(xs_batch)

            # get loss
            loss = torch.nn.MSELoss()(y_hat, ys_batch - y_hat_average)

            # backprop
            loss.backward()
            total_loss += loss.item()
        # update opt
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return total_loss/number_batches, norm.item()

    def test(self,
             example_xs: torch.tensor,
             example_ys: torch.tensor,
             xs: torch.tensor,
             ys: torch.tensor) -> float:
        assert xs[0].shape[1] == self.input_size, f"Input size of model '{self.input_size}' does not match input size of data '{xs.shape[1]}'"
        assert ys[0].shape[1] == self.output_size, f"Output size of model '{self.output_size}' does not match output size of data '{ys.shape[1]}'"

        with torch.no_grad():
            # due to the size of the data, we need to do gradient accumulation
            max_batch = 4  # max number of functions per gradient calculation
            number_batches = int(example_xs.shape[0] / max_batch)
            assert example_xs.shape[0] % max_batch == 0, f"example_xs.shape[0] ({example_xs.shape[0]}) must be divisible by max_batch ({max_batch})"

            total_loss = 0
            for batch_number in range(number_batches):
                # get batches
                xs_batch = xs[batch_number * max_batch: (batch_number + 1) * max_batch]
                ys_batch = ys[batch_number * max_batch: (batch_number + 1) * max_batch]
                example_xs_batch = example_xs[batch_number * max_batch: (batch_number + 1) * max_batch]
                example_ys_batch = example_ys[batch_number * max_batch: (batch_number + 1) * max_batch]

                # get encodings from example data
                individual_encoding = self.model(example_xs_batch)
                assert individual_encoding.shape == (
                example_xs_batch.shape[0], example_xs_batch.shape[1], self.output_size * self.embed_size)
                individual_encoding = individual_encoding.reshape(individual_encoding.shape[0],
                                                                  individual_encoding.shape[1], self.output_size, -1)
                assert individual_encoding.shape == (
                example_xs_batch.shape[0], example_xs_batch.shape[1], self.output_size, self.embed_size)

                with torch.no_grad():
                    example_ys_average = self.average_function(example_xs_batch)

                encodings = torch.mean(individual_encoding * (example_ys_batch - example_ys_average).unsqueeze(-1), dim=1)
                assert encodings.shape == (example_xs_batch.shape[0], self.output_size, self.embed_size)

                # use encodings to make prediction
                train_individual_encodings = self.model(xs_batch)
                assert train_individual_encodings.shape == (
                xs_batch.shape[0], xs_batch.shape[1], self.output_size * self.embed_size)
                train_individual_encodings = train_individual_encodings.reshape(train_individual_encodings.shape[0],
                                                                                train_individual_encodings.shape[1],
                                                                                self.output_size, -1)
                assert train_individual_encodings.shape == (
                xs_batch.shape[0], xs_batch.shape[1], self.output_size, self.embed_size)
                y_hat = torch.sum(train_individual_encodings * encodings.unsqueeze(1), dim=-1)
                assert y_hat.shape == ys_batch.shape, f"y_hat is wrong shape, got {y_hat.shape}, expected {ys.shape}"

                with torch.no_grad():
                    y_hat_average = self.average_function(xs_batch)

                # get loss
                loss = torch.nn.MSELoss()(y_hat, ys_batch - y_hat_average)
                total_loss += loss.item()

            return total_loss / number_batches

    def forward_testing(self,
                example_xs: torch.tensor, # F x B1 x SA size
                example_ys: torch.tensor, # F x B1 x S size
                xs: torch.tensor # F X B2 x SA size
                ) -> torch.Tensor:
        assert example_xs.shape[-1] == self.input_size, f"Input size of model '{self.input_size}' does not match input size of data '{example_xs.shape[1]}'"
        assert example_ys.shape[-1] == self.output_size, f"Output size of model '{self.output_size}' does not match output size of data '{example_ys.shape[1]}'"
        assert xs.shape[-1] == self.input_size, f"Input size of model '{self.input_size}' does not match input size of data '{xs.shape[1]}'"

        output_ys = torch.zeros((xs.shape[0], xs.shape[1], self.output_size)).to(self.device)
        max_batches_at_once = 10
        numb_runs = example_xs.shape[0] // max_batches_at_once

        for batch in range(numb_runs):
            example_xs_batch = example_xs[batch * max_batches_at_once: (batch + 1) * max_batches_at_once]
            example_ys_batch = example_ys[batch * max_batches_at_once: (batch + 1) * max_batches_at_once]
            xs_batch = xs[batch * max_batches_at_once: (batch + 1) * max_batches_at_once]

            # get encodings from example data
            individual_encoding = self.model(example_xs_batch)
            individual_encoding = individual_encoding.reshape(individual_encoding.shape[0], individual_encoding.shape[1],self.output_size, -1)
            encodings = torch.mean(individual_encoding * example_ys_batch.unsqueeze(-1), dim=1)

            # use encodings to make prediction
            train_individual_encodings = self.model(xs_batch)
            train_individual_encodings = train_individual_encodings.reshape(train_individual_encodings.shape[0],
                                                                            train_individual_encodings.shape[1],
                                                                            self.output_size, -1)

            y_hat = torch.sum(train_individual_encodings * encodings.unsqueeze(1), dim=-1)
            output_ys[batch * max_batches_at_once: (batch + 1) * max_batches_at_once] = y_hat
        return output_ys

    def get_encodings(self, example_xs: torch.tensor, example_ys: torch.tensor) -> torch.Tensor:
        with torch.no_grad():
            # due to the size of the data, we need to do gradient accumulation
            max_batch = 1  # max number of functions per gradient calculation
            number_batches = int(example_xs.shape[0] / max_batch)
            assert example_xs.shape[0] % max_batch == 0, f"example_xs.shape[0] ({example_xs.shape[0]}) must be divisible by max_batch ({max_batch})"

            encodings_all = torch.zeros((example_xs.shape[0], self.output_size, self.embed_size)).to(self.device)
            for batch_number in range(number_batches):
                # get batches
                example_xs_batch = example_xs[batch_number * max_batch: (batch_number + 1) * max_batch]
                example_ys_batch = example_ys[batch_number * max_batch: (batch_number + 1) * max_batch]

                # get encodings from example data
                individual_encoding = self.model(example_xs_batch)
                assert individual_encoding.shape == (example_xs_batch.shape[0], example_xs_batch.shape[1], self.output_size * self.embed_size)
                individual_encoding = individual_encoding.reshape(individual_encoding.shape[0], individual_encoding.shape[1], self.output_size, -1)
                assert individual_encoding.shape == (example_xs_batch.shape[0], example_xs_batch.shape[1], self.output_size, self.embed_size)

                encodings = torch.mean(individual_encoding * example_ys_batch.unsqueeze(-1), dim=1)
                encodings_all[batch_number * max_batch: (batch_number + 1) * max_batch] = encodings
            return encodings_all
        
    def update_average_function(self, xs: torch.tensor,ys: torch.tensor):
        assert xs.shape[-1] == self.input_size, f"Input size of model '{self.input_size}' does not match input size of data '{xs.shape[1]}'"
        assert ys.shape[-1] == self.output_size, f"Output size of model '{self.output_size}' does not match output size of data '{ys.shape[1]}'"

        # backprop
        self.optimizer_average_function.zero_grad()
        max_batch = 4  # max number of functions per gradient calculation
        number_batches = int(xs.shape[0] / max_batch)
        assert xs.shape[0] % max_batch == 0, f"example_xs.shape[0] ({xs.shape[0]}) must be divisible by max_batch ({max_batch})"
        for batch_number in range(number_batches):
            xs_batch = xs[batch_number * max_batch: (batch_number + 1) * max_batch]
            ys_batch = ys[batch_number * max_batch: (batch_number + 1) * max_batch]

            # compute loss  
            y_hat = self.average_function(xs_batch)
            assert y_hat.shape == ys_batch.shape, f"y_hat is wrong shape, got {y_hat.shape}, expected {ys.shape}"
            loss = torch.nn.MSELoss()(y_hat, ys_batch)
            loss.backward()
        self.optimizer_average_function.step()
