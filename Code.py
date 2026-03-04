import torch
import torch.nn as nn

class CustomLayer(nn.Module):
    def __init__(self, in_f, out_f, num_conn=None, sparse=False):
        super().__init__()
        if sparse:
            self.c = nn.Parameter(torch.randn(out_f, num_conn) * 2)
        else:
            self.c = nn.Parameter(torch.randn(out_f, in_f) * 2)
        self.k = nn.Parameter(torch.ones_like(self.c))
        self.b = nn.Parameter(torch.zeros(out_f))
        self.sparse = sparse

    def forward(self, x):
        if self.sparse:
            x = x.squeeze(-1)[:, None, None]
        else:
            x = x.unsqueeze(1)
        c = self.c.unsqueeze(0)
        r = 1.0 / (1.0 + (x - c) ** 2)
        o = self.k * r / (r.sum(dim=-1, keepdim=True) + 1e-8)
        return o.sum(dim=-1) + self.b

class DeepModel(nn.Module):
    def __init__(self, layer_sizes, first_conn=6):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(CustomLayer(1, layer_sizes[0], first_conn, sparse=True))
        for i in range(1, len(layer_sizes)):
            self.layers.append(CustomLayer(layer_sizes[i-1], layer_sizes[i], sparse=False))
        self.out = CustomLayer(layer_sizes[-1], 1, sparse=False)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.out(x)

def train_model(x, y, epochs=200, lr=0.02, layer_sizes=[8, 4]):
    model = DeepModel(layer_sizes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    
    return model, loss.item()

if __name__ == "__main__":
    # Use
    x = torch.tensor([[0.2], [0.6], [1.0], [1.4], [1.8]])
    y = torch.tanh(x)/2 + 0.05 * torch.randn_like(x)

    model, final_loss = train_model(x, y, epochs=200) 
    print('Final loss:', final_loss)
