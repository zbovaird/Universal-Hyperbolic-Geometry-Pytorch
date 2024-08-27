import torch
import torch.nn as nn
import torch.nn.functional as F

class UHGTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__()

    def to_poincare_disk(self):
        x = self[..., :-1]
        t = self[..., -1:]
        return x / (t + 1)

    def mobius_add(self, other):
        a, b = self.to_poincare_disk(), other.to_poincare_disk()
        num = (1 + 2 * torch.sum(a * b, dim=-1, keepdim=True) + torch.sum(b**2, dim=-1, keepdim=True)) * a + (1 - torch.sum(a**2, dim=-1, keepdim=True)) * b
        den = 1 + 2 * torch.sum(a * b, dim=-1, keepdim=True) + torch.sum(a**2, dim=-1, keepdim=True) * torch.sum(b**2, dim=-1, keepdim=True)
        return num / den

    def mobius_mul(self, scalar):
        x = self.to_poincare_disk()
        norm = torch.norm(x, dim=-1, keepdim=True)
        return torch.tanh(scalar * torch.atanh(norm)) * x / norm

    def exp_map(self, base_point):
        v = self.to_poincare_disk()
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        return self.mobius_add(base_point, torch.tanh(v_norm / 2) * v / v_norm)

    def log_map(self, base_point):
        y = self.to_poincare_disk()
        diff = self.mobius_add(-base_point, y)
        diff_norm = torch.norm(diff, dim=-1, keepdim=True)
        return 2 * torch.atanh(diff_norm) * diff / diff_norm

    def parallel_transport(self, src, dst):
        x = self.to_poincare_disk()
        src_dst = src.mobius_add(-dst)
        norm_src_dst = torch.norm(src_dst, dim=-1, keepdim=True)
        return x + 2 * torch.sum(x * src_dst, dim=-1, keepdim=True) / norm_src_dst**2 * (src + dst)

def hyperbolic_mean(points):
    # Compute Euclidean mean
    mean = torch.mean(points, dim=0)
    # Project back to hyperbolic space
    return mean / torch.sqrt(1 + torch.sum(mean**2))

class HyperbolicReLU(nn.Module):
    def forward(self, x):
        return torch.where(x > 0, x, torch.zeros_like(x))

class HyperbolicBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(HyperbolicBatchNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_normalized + self.bias

class HyperbolicGraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(HyperbolicGraphSAGELayer, self).__init__()
        self.linear_neigh = nn.Linear(in_features, out_features)
        self.linear_self = nn.Linear(in_features, out_features)
        self.activation = HyperbolicReLU()

    def forward(self, x, adj):
        # Aggregate neighborhood features
        neigh_features = torch.matmul(adj, x)
        neigh_features = self.linear_neigh(neigh_features)

        # Transform self features
        self_features = self.linear_self(x)

        # Combine and activate
        combined = neigh_features + self_features
        return self.activation(combined)

class HyperbolicGraphSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super(HyperbolicGraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HyperbolicGraphSAGELayer(in_features, hidden_features))
        for _ in range(num_layers - 2):
            self.layers.append(HyperbolicGraphSAGELayer(hidden_features, hidden_features))
        self.layers.append(HyperbolicGraphSAGELayer(hidden_features, out_features))
        self.batch_norm = HyperbolicBatchNorm(out_features)

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return self.batch_norm(x)

class RiemannianOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(RiemannianOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
                p.data = p.data / torch.sqrt(1 + torch.sum(p.data**2, dim=-1, keepdim=True))

        return loss

def train_hyperbolic_graphsage(model, optimizer, criterion, features, adj, labels, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(features, adj)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create dummy data
    num_nodes = 1000
    num_features = 16
    num_classes = 5
    features = torch.randn(num_nodes, num_features)
    adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    labels = torch.randint(0, num_classes, (num_nodes,))

    # Initialize model
    model = HyperbolicGraphSAGE(num_features, 32, num_classes, num_layers=2)

    # Define optimizer and loss
    optimizer = RiemannianOptimizer(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_hyperbolic_graphsage(model, optimizer, criterion, features, adj, labels, epochs=100)

    print("Training complete!")
