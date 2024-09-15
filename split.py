import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from torch.utils.data import DataLoader, TensorDataset

# Example model architecture
class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        return self.layer(x)

class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.layer(x)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_model, server_model):
        self.client_model = client_model
        self.server_model = server_model
        self.optimizer = optim.SGD(self.client_model.parameters(), lr=0.01)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.client_model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.client_model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.client_model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.client_model.train()

        # Dummy data
        data = torch.randn(32, 28 * 28)
        labels = torch.randint(0, 10, (32,))

        output = self.client_model(data)
        output = output.detach().requires_grad_(True)

        # Send to server
        server_output = self.server_model(output)

        loss = nn.functional.nll_loss(server_output, labels)
        loss.backward()

        self.optimizer.step()

        return self.get_parameters(), len(data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.client_model.eval()

        # Dummy data
        data = torch.randn(32, 28 * 28)
        labels = torch.randint(0, 10, (32,))

        with torch.no_grad():
            output = self.client_model(data)
            server_output = self.server_model(output)
            loss = nn.functional.nll_loss(server_output, labels)

        return loss.item(), len(data), {}


class FlowerServer(fl.server.Server):
    def __init__(self, client_model, server_model):
        self.client_model = client_model
        self.server_model = server_model

    def aggregate_fit(self, rnd, results, failures):
        # Optionally aggregate client model parameters
        pass

    def aggregate_evaluate(self, rnd, results, failures):
        # Optionally aggregate evaluation results
        pass

    def fit_round(self, rnd, parameters, config):
        # Pass the parameters to clients, run training, etc.
        return super().fit_round(rnd, parameters, config)

    def evaluate_round(self, rnd, parameters, config):
        # Evaluate models on the server-side dataset
        return super().evaluate_round(rnd, parameters, config)

if __name__ == "__main__":
    # Instantiate models
    client_model = ClientModel()
    server_model = ServerModel()

    # Start Flower server
    print("server begin")
    # = FlowerServer(client_model, server_model)
    #fl.server.start_server(server_address="0.0.0.0:8080")
    print("client begin")
    # Start Flower client
    client = FlowerClient(client_model, server_model)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client,config=fl.server.ServerConfig(num_rounds=4))