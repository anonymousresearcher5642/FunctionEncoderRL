from typing import List, Tuple

from MultiSystemIdentification.Predictor import Predictor
import torch

class FE_orthonormalization(Predictor):
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.embed_size = embed_size
        raise Exception("THis is too memory hungry in practice")

    def train(self,
              example_xs: torch.tensor,
              example_ys: torch.tensor,
              xs: torch.tensor,
              ys: torch.tensor) -> Tuple[float, float]:
        assert xs.shape[-1] == self.input_size, f"Input size of model '{self.input_size}' does not match input size of data '{xs.shape[1]}'"
        assert ys.shape[-1] == self.output_size, f"Output size of model '{self.output_size}' does not match output size of data '{ys.shape[1]}'"

        # backprop
        self.optimizer.zero_grad()

        # due to the size of the data, we need to do gradient accumulation
        max_batch = 1  # max number of functions per gradient calculation
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

            encodings = torch.mean(individual_encoding * example_ys_batch.unsqueeze(-1), dim=1)
            assert encodings.shape == (example_xs_batch.shape[0], self.output_size, self.embed_size)

            # add orthonormalization loss
            print(individual_encoding.shape)
            individual_encoding_inner_products = individual_encoding[0].unsqueeze(2) * individual_encoding[0].unsqueeze(3)
            print(individual_encoding_inner_products.shape)
            individual_encoding_inner_products = torch.sum(individual_encoding_inner_products, dim=0)
            # 2 x 17 x 100 x 100
            print(individual_encoding_inner_products.shape)
            # compare against identity matrix for the last 2 dimensions
            identity_matrix = torch.eye(self.embed_size).to(self.device)
            print(identity_matrix.shape)
            # compute loss
            loss_on = torch.nn.MSELoss()(individual_encoding_inner_products, identity_matrix.unsqueeze(0).expand(individual_encoding_inner_products.shape[0], -1, -1))

            # use encodings to make prediction
            train_individual_encodings = self.model(xs_batch)
            assert train_individual_encodings.shape == (xs_batch.shape[0], xs_batch.shape[1], self.output_size * self.embed_size)
            train_individual_encodings = train_individual_encodings.reshape(train_individual_encodings.shape[0], train_individual_encodings.shape[1], self.output_size, -1)
            assert train_individual_encodings.shape == (xs_batch.shape[0], xs_batch.shape[1], self.output_size, self.embed_size)
            y_hat = torch.sum(train_individual_encodings * encodings.unsqueeze(1), dim=-1)
            assert y_hat.shape == ys_batch.shape, f"y_hat is wrong shape, got {y_hat.shape}, expected {ys.shape}"

            # get loss
            loss = torch.nn.MSELoss()(y_hat, ys_batch)

            # backprop
            (loss + loss_on).backward()
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

                encodings = torch.mean(individual_encoding * example_ys_batch.unsqueeze(-1), dim=1)
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

                # get loss
                loss = torch.nn.MSELoss()(y_hat, ys_batch)
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