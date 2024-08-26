import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class HyperbolicMSELoss(nn.Module):
    def __init__(self, c=1.0):
        super(HyperbolicMSELoss, self).__init__()
        self.c = torch.tensor(c, dtype=torch.float32)

    def forward(self, pred, target):
        return self.hyperbolic_distance(pred, target).pow(2).mean()

    def hyperbolic_distance(self, x, y):
        c = self.c.to(x.device)
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-15, max=1-1e-5)
        y_norm = torch.norm(y, dim=-1, keepdim=True).clamp(min=1e-15, max=1-1e-5)
        
        num = torch.norm(x - y, dim=-1).clamp(min=1e-15)
        den = ((1 - c * x_norm**2) * (1 - c * y_norm**2)).clamp(min=1e-15)
        
        arg = (torch.sqrt(c) * num / torch.sqrt(den)).clamp(min=-1+1e-7, max=1-1e-7)
        return (2 / torch.sqrt(c)) * torch.atanh(arg)

class HyperbolicNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(HyperbolicNeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.001, 0.001)
        nn.init.xavier_uniform_(self.hidden.weight, gain=0.01)
        nn.init.xavier_uniform_(self.output.weight, gain=0.01)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)  # Average the embeddings for each sample
        x = torch.tanh(self.hidden(x))
        x = torch.tanh(self.output(x))
        return x

def train_hyperbolic_nn(model, criterion, optimizer, train_data, num_epochs=100):
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf loss encountered at epoch {epoch}")
                continue
            loss.backward()
            
            # Clip gradient values
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_data)}")

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define hyperparameters
    vocab_size = 1000
    embed_dim = 50
    hidden_dim = 20
    output_dim = 3
    sequence_length = 5

    # Create a hyperbolic neural network
    model = HyperbolicNeuralNetwork(vocab_size, embed_dim, hidden_dim, output_dim)

    # Define loss function and optimizer
    criterion = HyperbolicMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Reduced learning rate

    # Create some dummy training data
    num_samples = 100
    inputs = torch.randint(0, vocab_size, (num_samples, sequence_length))
    targets = torch.rand((num_samples, output_dim))
    targets = targets / (1 + torch.norm(targets, dim=-1, keepdim=True))  # Project to Poincaré ball
    train_data = [(inputs[i:i+10], targets[i:i+10]) for i in range(0, num_samples, 10)]

    # Train the model
    train_hyperbolic_nn(model, criterion, optimizer, train_data, num_epochs=100)

    # Test the trained model
    test_input = torch.LongTensor([[0, 2, 5, 1, 4]])
    output = model(test_input)

    print("Hyperbolic Neural Network Output:")
    print(output)

    print("\nModel parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: shape {param.data.shape}")
