import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import SimpleCNN
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameters ---
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

def train():
    # 1. Load Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Initialize Model
    model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    print(model)
    
    # 3. Define Loss Function and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),
                         lr=LEARNING_RATE, 
                         weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                    mode='min', 
                                                    factor=0.5, 
                                                    patience=3, 
                                                    verbose=True)

    # Initialize history lists
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    print("\n--- Memulai Training ---")
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training phase
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels.float())
                val_running_loss += loss.item()
                
                predicted = (outputs > 0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    print("--- Training Selesai ---")
    
    # Tampilkan plot
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)

    # Visualisasi prediksi
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    train()