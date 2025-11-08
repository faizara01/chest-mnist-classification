import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model_resnet2 import DeeperResNet
from utils import plot_training_history, visualize_random_val_predictions

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0001

# Optimizer configurations
optimizer_configs = {
    'adam': {
        'optimizer': optim.Adam,
        'params': {'lr': 0.001, 'betas': (0.9, 0.999)}
    },
    'sgd': {
        'optimizer': optim.SGD,
        'params': {'lr': 0.01, 'momentum': 0.9}
    },
    'rmsprop': {
        'optimizer': optim.RMSprop,
        'params': {'lr': 0.0001, 'momentum': 0.9}
    }
}

def train_resnet(optimizer_name='adam'):
    # Load Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeeperResNet(in_channels=in_channels, num_classes=num_classes).to(device)
    print(model)
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    opt_config = optimizer_configs[optimizer_name]
    optimizer = opt_config['optimizer'](model.parameters(), **opt_config['params'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # History tracking
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\n--- Starting Training with {optimizer_name.upper()} ---")
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.float())
                
                val_loss += loss.item()
                predicted = (outputs > 0).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    print(f"--- Training Complete with {optimizer_name.upper()} ---")
    return train_losses, val_losses, train_accs, val_accs

if __name__ == '__main__':
    for opt in ['adam', 'sgd', 'rmsprop']:
        histories = train_resnet(optimizer_name=opt)
        plot_training_history(*histories, title=f'Training History - {opt.upper()}')