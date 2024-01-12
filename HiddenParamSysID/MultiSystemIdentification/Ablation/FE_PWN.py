from typing import List, Tuple

from MultiSystemIdentification.Predictor import Predictor
import torch

class FE_PWN(Predictor):
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

            # DO point wise normalization
            individual_encoding = individual_encoding / torch.sum(individual_encoding**2, dim=-1, keepdim=True)**0.5

            encodings = torch.mean(individual_encoding * example_ys_batch.unsqueeze(-1), dim=1)
            assert encodings.shape == (example_xs_batch.shape[0], self.output_size, self.embed_size)

            # compute cos similiarity between all encodings
            # cos_sim = torch.nn.CosineSimilarity(dim=-1)
            # cos_sim_matrix = cos_sim(encodings.unsqueeze(1), encodings.unsqueeze(0))
            #
            # # compute stats on ys_concat such as max, min, mean, std dev
            # ys_maxes = torch.max(torch.max(ys_concat, dim=1)[0], dim=0)[0]
            # ys_mins = torch.min(torch.min(ys_concat, dim=1)[0], dim=0)[0]
            # ys_means = torch.mean(torch.mean(ys_concat, dim=1), dim=0)
            # ys_std_devs = torch.mean(torch.std(ys_concat, dim=1), dim=0)

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

                # DO point wise normalization
                individual_encoding = individual_encoding / torch.sum(individual_encoding**2, dim=-1, keepdim=True)**0.5

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
                # DO point wise normalization
                individual_encoding = individual_encoding / torch.sum(individual_encoding**2, dim=-1, keepdim=True)**0.5

                encodings = torch.mean(individual_encoding * example_ys_batch.unsqueeze(-1), dim=1)
                encodings_all[batch_number * max_batch: (batch_number + 1) * max_batch] = encodings
            return encodings_all