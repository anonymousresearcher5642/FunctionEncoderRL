import os
import warnings
from typing import Tuple, Any

import torch
from torch import Tensor

from MultiSystemIdentification.Predictor import Predictor

class Transformer(Predictor):
    def __init__(self, input_size, output_size, device, d_model=200):
        warnings.warn(" I thought this implementation would be better than the other, but its performance is slightly worse. Use the other for the paper's results. ")
        super().__init__(input_size, output_size, device)
        self.transformer = torch.nn.Transformer(d_model=d_model,
                                                nhead=4,
                                                num_encoder_layers=4,
                                                num_decoder_layers=4,
                                                dim_feedforward=d_model,
                                                dropout=0.0,
                                                batch_first=True).to(device)
        self.encoder_examples = torch.nn.Sequential( # encodes inputs (state x action x next_state)
            torch.nn.Linear(input_size + output_size, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
        ).to(device)
        self.encoder_current_transition = torch.nn.Sequential( # encodes the current inputs (state x action)
            torch.nn.Linear(input_size, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
        ).to(device)
        self.decoder = torch.nn.Sequential( # outputs the next state
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, output_size),
        ).to(device)
        self.optimizer = torch.optim.Adam([ *self.transformer.parameters(),
                                                   *self.encoder_examples.parameters(),
                                                   *self.encoder_current_transition.parameters(),
                                                   *self.decoder.parameters()], lr=1e-3)
        self.d_model = d_model

    def num_params(self) -> int:
        assert self.transformer and self.encoder_examples and self.encoder_current_transition and self.decoder
        return sum(p.numel() for p in self.transformer.parameters() if p.requires_grad) + \
               sum(p.numel() for p in self.encoder_examples.parameters() if p.requires_grad) + \
               sum(p.numel() for p in self.encoder_current_transition.parameters() if p.requires_grad) + \
               sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

    def forward(self,
                example_xs: torch.tensor, # F x B1 x SA size
                example_ys: torch.tensor, # F x B1 x S size
                xs: torch.tensor # F X B2 x SA size
                ) -> Tensor:
        assert example_xs.shape[0] == 1, "Multiple functions at once not supported for transformers yet. "
        assert example_ys.shape[0] == 1, "Multiple functions at once not supported for transformers yet. "
        assert xs.shape[0] == 1, "Multiple functions at once not supported for transformers yet. "

        # convert all data to encodings
        examples = torch.cat((example_xs, example_ys), dim=-1)
        example_input_encodings = self.encoder_examples(examples) # F x B1 x D size
        # example_input_encodings = self.encoder_inputs(example_xs) # F x B1 x D size
        # example_output_encodings = self.encoder_outputs(example_ys) # F x B1 x D size
        real_input_encoding = self.encoder_current_transition(xs) # F x B2 x D size
        # assert example_input_encodings.shape == (example_xs.shape[0], example_xs.shape[1], self.d_model)
        # assert example_output_encodings.shape == (example_ys.shape[0], example_ys.shape[1], self.d_model)
        # assert real_input_encoding.shape == (xs.shape[0], xs.shape[1], self.d_model)

        # convert example encodings to something we can feed into model
        # combined_encoder_inputs = torch.zeros((example_input_encodings.shape[0], 2 * example_input_encodings.shape[1], self.d_model)).to(self.device)
        # combined_encoder_inputs[:, 0::2, :] = example_input_encodings
        # combined_encoder_inputs[:, 1::2, :] = example_output_encodings
        # combined_encoder_inputs = combined_encoder_inputs.expand(xs.shape[1], combined_encoder_inputs.shape[1], combined_encoder_inputs.shape[2])
        # assert combined_encoder_inputs.shape == (xs.shape[1], 2 * example_input_encodings.shape[1], self.d_model)

        # convert real encodings to something we can feed into model
        real_input_encoding = real_input_encoding[0]
        real_input_encoding = real_input_encoding.unsqueeze(1)
        assert real_input_encoding.shape == (xs.shape[1], 1, self.d_model)

        example_input_encodings = example_input_encodings.repeat(real_input_encoding.shape[0], 1, 1)
        assert example_input_encodings.shape == (real_input_encoding.shape[0], example_xs.shape[1], self.d_model)

        output_embedding = self.transformer(example_input_encodings, real_input_encoding)
        output = self.decoder(output_embedding)
        return output

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
            max_batch = 1 # max number of functions per gradient calculation
            number_batches = int(example_xs.shape[0] / max_batch)
            assert example_xs.shape[0] % max_batch == 0, f"example_xs.shape[0] ({example_xs.shape[0]}) must be divisible by max_batch ({max_batch})"

            # due to memory hogging of transormers, we need to do even more gradient acucmulation
            max_predictions = 200

            # max samples per batch
            max_samples = 200

            total_loss = 0
            for batch_number in range(number_batches):
                # grabs data form one function only
                example_ys_batch = example_ys[batch_number * max_batch: (batch_number + 1) * max_batch]
                example_xs_batch = example_xs[batch_number * max_batch: (batch_number + 1) * max_batch]
                xs_batch = xs[batch_number * max_batch: (batch_number + 1) * max_batch]
                ys_batch = ys[batch_number * max_batch: (batch_number + 1) * max_batch]

                # we also need to reduce the number of examples since transformers use so much memory
                permutation = torch.randperm(example_xs_batch.shape[1])
                example_xs_batch = example_xs_batch[:, permutation[:max_samples]]
                example_ys_batch = example_ys_batch[:, permutation[:max_samples]]

                # also permute the xs and ys
                permutation = torch.randperm(xs_batch.shape[1])
                xs_batch = xs_batch[:, permutation[:max_predictions]]
                ys_batch = ys_batch[:, permutation[:max_predictions]]

                # compute loss
                y_hat_batch = self.forward(example_xs_batch, example_ys_batch, xs_batch)
                y_hat_batch = y_hat_batch.view(1, -1, self.output_size)
                assert y_hat_batch.shape == ys_batch.shape, f"y_hat is wrong shape, got {y_hat_batch.shape}, expected {ys_batch.shape}"
                loss = torch.nn.MSELoss()(y_hat_batch, ys_batch)

                loss.backward()
                total_loss += loss.item()

            self.optimizer.step()
            return total_loss/number_batches, None

    def test(self,
                example_xs: torch.tensor,
                example_ys: torch.tensor,
                xs: torch.tensor,
                ys: torch.tensor) -> float:
        with torch.no_grad():
            # due to the size of the data, we need to do gradient accumulation
            max_batch = 1  # max number of functions per gradient calculation
            number_batches = int(example_xs.shape[0] / max_batch)
            assert example_xs.shape[0] % max_batch == 0, f"example_xs.shape[0] ({example_xs.shape[0]}) must be divisible by max_batch ({max_batch})"

            # due to memory hogging of transormers, we need to do even more gradient acucmulation
            max_predictions = 200

            # max samples per batch
            max_samples = 200

            total_loss = 0
            for batch_number in range(number_batches):
                # grabs data form one function only
                example_ys_batch = example_ys[batch_number * max_batch: (batch_number + 1) * max_batch]
                example_xs_batch = example_xs[batch_number * max_batch: (batch_number + 1) * max_batch]
                xs_batch = xs[batch_number * max_batch: (batch_number + 1) * max_batch]
                ys_batch = ys[batch_number * max_batch: (batch_number + 1) * max_batch]

                # we also need to reduce the number of examples since transformers use so much memory
                permutation = torch.randperm(example_xs_batch.shape[1])
                example_xs_batch = example_xs_batch[:, permutation[:max_samples]]
                example_ys_batch = example_ys_batch[:, permutation[:max_samples]]

                # also permute the xs and ys
                permutation = torch.randperm(xs_batch.shape[1])
                xs_batch = xs_batch[:, permutation[:max_predictions]]
                ys_batch = ys_batch[:, permutation[:max_predictions]]

                # compute loss
                y_hat_batch = self.forward(example_xs_batch, example_ys_batch, xs_batch)
                y_hat_batch = y_hat_batch.view(1, -1, self.output_size)
                assert y_hat_batch.shape == ys_batch.shape, f"y_hat is wrong shape, got {y_hat_batch.shape}, expected {ys_batch.shape}"
                loss = torch.nn.MSELoss()(y_hat_batch, ys_batch)
                total_loss += loss.item()

            return total_loss / number_batches

    def save(self, path):
        torch.save(self.transformer.state_dict(), os.path.join(path, "transformer.pt"))
        torch.save(self.encoder_examples.state_dict(), os.path.join(path, "encoder_examples.pt"))
        torch.save(self.encoder_current_transition.state_dict(), os.path.join(path, "encoder_current_transition.pt"))
        torch.save(self.decoder.state_dict(), os.path.join(path, "decoder.pt"))

    def load(self, path):
        self.transformer.load_state_dict(torch.load(os.path.join(path, "transformer.pt")))
        self.encoder_examples.load_state_dict(torch.load(os.path.join(path, "encoder_examples.pt")))
        self.encoder_current_transition.load_state_dict(torch.load(os.path.join(path, "encoder_current_transition.pt")))
        self.decoder.load_state_dict(torch.load(os.path.join(path, "decoder.pt")))

    # this automatically breaks it into batches if it is not batched
    def forward_testing(self,
                example_xs: torch.tensor, # F x B1 x SA size
                example_ys: torch.tensor, # F x B1 x S size
                xs: torch.tensor # F X B2 x SA size
                ) -> Tensor:

        num_batches = xs.shape[0]
        num_examples_at_once = 1000
        max_samples = 200 # we can only use 200 example datapoints because transformers hog memory
        example_ys = example_ys[:, :max_samples, :]
        example_xs = example_xs[:, :max_samples, :]

        # prepare outputs
        output_ys = torch.zeros((num_batches, xs.shape[1], self.output_size)).to(self.device)
        for batch in range(num_batches):
            xs_batch = xs[batch].unsqueeze(0)
            example_xs_batch = example_xs[batch].unsqueeze(0)
            example_ys_batch = example_ys[batch].unsqueeze(0)
            num_examples_at_once = min(num_examples_at_once, xs_batch.shape[1])
            number_runs = xs_batch.shape[1] // num_examples_at_once
            for run in range(number_runs):
                # get batches
                xs_batch_batch = xs_batch[:, run * num_examples_at_once: (run + 1) * num_examples_at_once]

                # convert all data to encodings
                examples = torch.cat((example_xs_batch, example_ys_batch), dim=-1)
                example_input_encodings = self.encoder_examples(examples) # F x B1 x D size

                # example_input_encodings = self.encoder_inputs(example_xs_batch) # F x B1 x D size
                # example_output_encodings = self.encoder_outputs(example_ys_batch) # F x B1 x D size
                real_input_encoding = self.encoder_inputs(xs_batch_batch) # F x B2 x D size
                assert example_input_encodings.shape == (example_xs_batch.shape[0], example_xs_batch.shape[1], self.d_model)
                # assert example_output_encodings.shape == (example_ys_batch.shape[0], example_ys_batch.shape[1], self.d_model)
                assert real_input_encoding.shape == (xs_batch_batch.shape[0], xs_batch_batch.shape[1], self.d_model)

                # convert example encodings to something we can feed into model
                # combined_encoder_inputs = torch.zeros((example_input_encodings.shape[0], 2 * example_input_encodings.shape[1], self.d_model)).to(self.device)
                # combined_encoder_inputs[:, 0::2, :] = example_input_encodings
                # combined_encoder_inputs[:, 1::2, :] = example_output_encodings
                # combined_encoder_inputs = combined_encoder_inputs.expand(xs_batch_batch.shape[1], combined_encoder_inputs.shape[1], combined_encoder_inputs.shape[2])
                # assert combined_encoder_inputs.shape == (xs_batch_batch.shape[1], 2 * example_input_encodings.shape[1], self.d_model)

                # convert real encodings to something we can feed into model
                real_input_encoding = real_input_encoding[0]
                real_input_encoding = real_input_encoding.unsqueeze(1)
                assert real_input_encoding.shape == (xs_batch_batch.shape[1], 1, self.d_model)

                output_embedding = self.transformer(example_input_encodings, real_input_encoding)
                output = self.decoder(output_embedding).view(1, -1, self.output_size)

                output_ys[batch, run * num_examples_at_once: (run + 1) * num_examples_at_once, :] = output
        return output_ys