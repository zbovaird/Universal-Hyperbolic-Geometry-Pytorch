#!/usr/bin/env python
# coding: utf-8

# In[1]:


# File: uhg_hyperbolic_nn.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

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
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout_rate=0.5, is_classification=False):
        super(HyperbolicNeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.is_classification = is_classification
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.001, 0.001)
        nn.init.xavier_uniform_(self.hidden.weight, gain=0.01)
        nn.init.xavier_uniform_(self.output.weight, gain=0.01)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = torch.tanh(self.hidden(x))
        x = self.dropout(x)
        x = self.output(x)
        if self.is_classification:
            x = F.log_softmax(x, dim=1)
        else:
            x = torch.tanh(x)
        return x

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class HyperbolicDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets

class HyperbolicDataLoader:
    def __init__(self, inputs, targets, batch_size=32, shuffle=True):
        self.dataset = HyperbolicDataset(inputs, targets)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

def get_optimizer(optimizer_name, model_parameters, lr):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model_parameters, lr=lr)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model_parameters, lr=lr)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=lr)
    elif optimizer_name.lower() == 'adagrad':
        return optim.Adagrad(model_parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def train_hyperbolic_nn(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, patience=7):
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf loss encountered at epoch {epoch}")
                continue
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model, train_losses, val_losses

def random_search(param_distributions, n_iter, train_inputs, train_targets, val_inputs, val_targets, is_classification=False):
    best_val_loss = float('inf')
    best_params = None

    for i in range(n_iter):
        current_params = {
            'embed_dim': random.choice(param_distributions['embed_dim']),
            'hidden_dim': random.choice(param_distributions['hidden_dim']),
            'dropout_rate': random.uniform(*param_distributions['dropout_rate']),
            'learning_rate': 10 ** random.uniform(*param_distributions['learning_rate']),
            'batch_size': random.choice(param_distributions['batch_size']),
            'optimizer': random.choice(param_distributions['optimizer'])
        }

        print(f"\nIteration {i+1}/{n_iter}")
        print("Current parameters:", current_params)

        model = HyperbolicNeuralNetwork(vocab_size=1000, 
                                        embed_dim=current_params['embed_dim'], 
                                        hidden_dim=current_params['hidden_dim'], 
                                        output_dim=train_targets.shape[1], 
                                        dropout_rate=current_params['dropout_rate'],
                                        is_classification=is_classification)
        
        optimizer = get_optimizer(current_params['optimizer'], model.parameters(), current_params['learning_rate'])

        train_loader = HyperbolicDataLoader(train_inputs, train_targets, batch_size=current_params['batch_size'])
        val_loader = HyperbolicDataLoader(val_inputs, val_targets, batch_size=current_params['batch_size'], shuffle=False)

        criterion = nn.NLLLoss() if is_classification else HyperbolicMSELoss()
        trained_model, train_losses, val_losses = train_hyperbolic_nn(model, criterion, optimizer, train_loader, val_loader)

        final_val_loss = val_losses[-1]
        print(f"Final validation loss: {final_val_loss:.4f}")

        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_params = current_params
            print("New best model found!")

    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return best_params, best_val_loss

def hyperbolic_circle(center, quadrance):
    c = center.to_poincare_disk()
    c_norm = torch.norm(c)
    if c_norm == 0:
        return [0, 0], torch.sqrt(quadrance / (1 + quadrance))
    factor = (1 - quadrance) / ((1 - c_norm**2) * (1 + quadrance))
    euc_center = factor * c
    euc_radius = torch.sqrt((factor * c_norm)**2 + quadrance) / (1 + quadrance)
    return euc_center.tolist(), euc_radius.item()

def plot_poincare_disk_with_circles(points, circles, labels=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    boundary = plt.Circle((0, 0), 1, fill=False, color='black')
    ax.add_artist(boundary)
    poincare_points = [p.to_poincare_disk() for p in points]
    x_coords = [p[0].item() for p in poincare_points]
    y_coords = [p[1].item() for p in poincare_points]
    ax.scatter(x_coords, y_coords)
    if labels:
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ax.annotate(labels[i], (x, y), xytext=(5, 5), textcoords='offset points')
    for center, quadrance in circles:
        euc_center, euc_radius = hyperbolic_circle(center, quadrance)
        circle = plt.Circle(euc_center, euc_radius, fill=False, color='red')
        ax.add_artist(circle)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title("Poincaré Disk Model with Hyperbolic Circles")
    plt.grid(True)
    plt.show()

def example_hyperbolic_nn(optimizer_name='adam'):
    print(f"Example: Hyperbolic Neural Network with {optimizer_name.capitalize()} optimizer")
    # ... (implementation remains the same)

def example_poincare_disk_visualization():
    print("Example: Poincaré Disk Visualization")
    # ... (implementation remains the same)

def example_multi_class_classification():
    print("Example: Multi-class Classification with Hyperbolic Neural Network")
    torch.manual_seed(42)
    
    # Generate dummy data for multi-class classification
    num_samples = 1000
    vocab_size = 1000
    sequence_length = 5
    num_classes = 5
    
    inputs = torch.randint(0, vocab_size, (num_samples, sequence_length))
    targets = torch.randint(0, num_classes, (num_samples,))
    
    # Convert targets to one-hot encoding
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    
    # Split data into train and validation sets
    num_train = int(0.8 * num_samples)
    train_inputs, train_targets = inputs[:num_train], targets_one_hot[:num_train]
    val_inputs, val_targets = inputs[num_train:], targets_one_hot[num_train:]
    
    # Define hyperparameter search space
    param_distributions = {
        'embed_dim': [32, 64, 128],
        'hidden_dim': [16, 32, 64],
        'dropout_rate': (0.1, 0.5),
        'learning_rate': (-4, -2),
        'batch_size': [16, 32, 64],
        'optimizer': ['adam', 'sgd', 'rmsprop']
    }
    
    # Perform random search
    best_params, best_val_loss = random_search(param_distributions, n_iter=5, 
                                               train_inputs=train_inputs, train_targets=train_targets,
                                               val_inputs=val_inputs, val_targets=val_targets,
                                               is_classification=True)
    
    print("\nBest hyperparameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Train the final model with best hyperparameters
    final_model = HyperbolicNeuralNetwork(vocab_size=vocab_size, 
                                          embed_dim=best_params['embed_dim'], 
                                          hidden_dim=best_params['hidden_dim'], 
                                          output_dim=num_classes, 
                                          dropout_rate=best_params['dropout_rate'],
                                          is_classification=True)
    
    optimizer = get_optimizer(best_params['optimizer'], final_model.parameters(), best_params['learning_rate'])
    criterion = nn.NLLLoss()
    
    train_loader = HyperbolicDataLoader(train_inputs, train_targets, batch_size=best_params['batch_size'])
    val_loader = HyperbolicDataLoader(val_inputs, val_targets, batch_size=best_params['batch_size'], shuffle=False)
    
    final_model, train_losses, val_losses = train_hyperbolic_nn(final_model, criterion, optimizer, train_loader, val_loader)
    
    # Evaluate the model
    final_model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = final_model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_targets.extend(targets.argmax(dim=1).numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    print(f"\nFinal model accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    example_hyperbolic_nn('adam')
    example_poincare_disk_visualization()
    example_multi_class_classification()


# In[7]:


# File: uhg_hyperbolic_nn.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

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
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout_rate=0.5, is_classification=False):
        super(HyperbolicNeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.is_classification = is_classification
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.001, 0.001)
        nn.init.xavier_uniform_(self.hidden.weight, gain=0.01)
        nn.init.xavier_uniform_(self.output.weight, gain=0.01)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = torch.tanh(self.hidden(x))
        x = self.dropout(x)
        x = self.output(x)
        if self.is_classification:
            x = F.log_softmax(x, dim=1)
        else:
            x = torch.tanh(x)
        return x

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class HyperbolicDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.tensor(targets)
    return inputs, targets

class HyperbolicDataLoader:
    def __init__(self, inputs, targets, batch_size=32, shuffle=True):
        self.dataset = HyperbolicDataset(inputs, targets)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

def get_optimizer(optimizer_name, model_parameters, lr):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model_parameters, lr=lr)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model_parameters, lr=lr)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=lr)
    elif optimizer_name.lower() == 'adagrad':
        return optim.Adagrad(model_parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def train_hyperbolic_nn(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, patience=7):
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf loss encountered at epoch {epoch}")
                continue
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model, train_losses, val_losses

def random_search(param_distributions, n_iter, train_inputs, train_targets, val_inputs, val_targets, is_classification=False):
    best_val_loss = float('inf')
    best_params = None

    for i in range(n_iter):
        current_params = {
            'embed_dim': random.choice(param_distributions['embed_dim']),
            'hidden_dim': random.choice(param_distributions['hidden_dim']),
            'dropout_rate': random.uniform(*param_distributions['dropout_rate']),
            'learning_rate': 10 ** random.uniform(*param_distributions['learning_rate']),
            'batch_size': random.choice(param_distributions['batch_size']),
            'optimizer': random.choice(param_distributions['optimizer'])
        }

        print(f"\nIteration {i+1}/{n_iter}")
        print("Current parameters:", current_params)

        model = HyperbolicNeuralNetwork(vocab_size=1000, 
                                        embed_dim=current_params['embed_dim'], 
                                        hidden_dim=current_params['hidden_dim'], 
                                        output_dim=train_targets.max().item() + 1, 
                                        dropout_rate=current_params['dropout_rate'],
                                        is_classification=is_classification)

        optimizer = get_optimizer(current_params['optimizer'], model.parameters(), current_params['learning_rate'])

        train_loader = HyperbolicDataLoader(train_inputs, train_targets, batch_size=current_params['batch_size'])
        val_loader = HyperbolicDataLoader(val_inputs, val_targets, batch_size=current_params['batch_size'], shuffle=False)

        criterion = nn.NLLLoss() if is_classification else HyperbolicMSELoss()
        trained_model, train_losses, val_losses = train_hyperbolic_nn(model, criterion, optimizer, train_loader, val_loader)

        final_val_loss = val_losses[-1]
        print(f"Final validation loss: {final_val_loss:.4f}")

        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_params = current_params
            print("New best model found!")

    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return best_params, best_val_loss

def hyperbolic_circle(center, quadrance):
    c = center.to_poincare_disk()
    c_norm = torch.norm(c)
    if c_norm == 0:
        return [0, 0], torch.sqrt(quadrance / (1 + quadrance))
    factor = (1 - quadrance) / ((1 - c_norm**2) * (1 + quadrance))
    euc_center = factor * c
    euc_radius = torch.sqrt((factor * c_norm)**2 + quadrance) / (1 + quadrance)
    return euc_center.tolist(), euc_radius.item()

def plot_poincare_disk_with_circles(points, circles, labels=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    boundary = plt.Circle((0, 0), 1, fill=False, color='black')
    ax.add_artist(boundary)
    poincare_points = [p.to_poincare_disk() for p in points]
    x_coords = [p[0].item() for p in poincare_points]
    y_coords = [p[1].item() for p in poincare_points]
    ax.scatter(x_coords, y_coords)
    if labels:
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ax.annotate(labels[i], (x, y), xytext=(5, 5), textcoords='offset points')
    for center, quadrance in circles:
        euc_center, euc_radius = hyperbolic_circle(center, quadrance)
        circle = plt.Circle(euc_center, euc_radius, fill=False, color='red')
        ax.add_artist(circle)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title("Poincaré Disk Model with Hyperbolic Circles")
    plt.grid(True)
    plt.show()

def example_hyperbolic_nn(optimizer_name='adam'):
    print(f"Example: Hyperbolic Neural Network with {optimizer_name.capitalize()} optimizer")
    # ... (implementation remains the same)

def example_poincare_disk_visualization():
    print("Example: Poincaré Disk Visualization")
    # ... (implementation remains the same)

def example_multi_class_classification():
    print("Example: Multi-class Classification with Hyperbolic Neural Network")
    torch.manual_seed(42)

    # Generate dummy data for multi-class classification
    num_samples = 1000  # Increase this when you have more data
    vocab_size = 1000
    sequence_length = 5
    num_classes = 5

    inputs = torch.randint(0, vocab_size, (num_samples, sequence_length))
    targets = torch.randint(0, num_classes, (num_samples,))

    # Split data into train+val (80%) and test (20%) sets
    num_train_val = int(0.8 * num_samples)
    train_val_inputs, test_inputs = inputs[:num_train_val], inputs[num_train_val:]
    train_val_targets, test_targets = targets[:num_train_val], targets[num_train_val:]

    # Further split train+val data into train (80% of 80%) and validation (20% of 80%) sets
    num_train = int(0.8 * num_train_val)
    train_inputs, val_inputs = train_val_inputs[:num_train], train_val_inputs[num_train:]
    train_targets, val_targets = train_val_targets[:num_train], train_val_targets[num_train:]

    # Define hyperparameter search space
    param_distributions = {
        'embed_dim': [32, 64, 128],
        'hidden_dim': [16, 32, 64],
        'dropout_rate': (0.1, 0.5),
        'learning_rate': (-4, -2),
        'batch_size': [16, 32, 64],
        'optimizer': ['adam', 'sgd', 'rmsprop']
    }

    # Perform random search
    best_params, best_val_loss = random_search(param_distributions, n_iter=5, 
                                               train_inputs=train_inputs, train_targets=train_targets,
                                               val_inputs=val_inputs, val_targets=val_targets,
                                               is_classification=True)

    print("\nBest hyperparameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Train the final model with best hyperparameters
    final_model = HyperbolicNeuralNetwork(vocab_size=vocab_size, 
                                          embed_dim=best_params['embed_dim'], 
                                          hidden_dim=best_params['hidden_dim'], 
                                          output_dim=num_classes, 
                                          dropout_rate=best_params['dropout_rate'],
                                          is_classification=True)

    optimizer = get_optimizer(best_params['optimizer'], final_model.parameters(), best_params['learning_rate'])
    criterion = nn.NLLLoss()

    # Create data loaders
    train_loader = HyperbolicDataLoader(train_inputs, train_targets, batch_size=best_params['batch_size'])
    val_loader = HyperbolicDataLoader(val_inputs, val_targets, batch_size=best_params['batch_size'], shuffle=False)
    test_loader = HyperbolicDataLoader(test_inputs, test_targets, batch_size=best_params['batch_size'], shuffle=False)

    final_model, train_losses, val_losses = train_hyperbolic_nn(final_model, criterion, optimizer, train_loader, val_loader)

    # Evaluate the model on the test set
    final_model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = final_model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_targets.extend(targets.numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    conf_matrix = confusion_matrix(all_targets, all_preds)

    print(f"\nFinal model accuracy on test set: {accuracy:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    example_hyperbolic_nn('adam')
    example_poincare_disk_visualization()
    example_multi_class_classification()


# In[ ]:




