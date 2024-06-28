import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SDFDataset
from model import SDFMLP
import argparse

def compute_gradient(inputs, outputs):
    gradients = torch.autograd.grad(outputs=outputs, inputs=inputs,
                                    grad_outputs=torch.ones_like(outputs),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    return gradients

def train_sdf_model(data_dir, model_save_path, num_epochs=100, batch_size=64, learning_rate=1e-4, use_fourier=False):
    # Load dataset
    dataset = SDFDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = SDFMLP(use_fourier=use_fourier)
    sdf_criterion = nn.MSELoss()
    grad_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Check if GPU is available and move model to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_grad_loss = 0.0

        # Using tqdm for progress bar
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for batch in tepoch:
                sdf_points = batch['sdf_points'].to(device).requires_grad_(True)
                sdf_values = batch['sdf_values'].to(device)
                sdf_gradients = batch['sdf_gradients'].to(device)

                optimizer.zero_grad()
                
                # Forward pass
                sdf_predictions = model(sdf_points)
                
                # Compute SDF loss
                sdf_loss = sdf_criterion(sdf_predictions.squeeze(), sdf_values)
                
                # Compute predicted gradients
                predicted_gradients = compute_gradient(sdf_points, sdf_predictions)
                
                # Compute gradient loss
                grad_loss = grad_criterion(predicted_gradients, sdf_gradients)
                
                # Total loss
                # loss = sdf_loss + grad_loss
                loss = sdf_loss
                loss.backward()
                optimizer.step()

                running_loss += sdf_loss.item() * sdf_points.size(0)
                running_grad_loss += grad_loss.item() * sdf_points.size(0)
                tepoch.set_postfix(sdf_loss=sdf_loss.item(), grad_loss=grad_loss.item(), total_loss=loss.item())

        epoch_sdf_loss = running_loss / len(dataset)
        epoch_grad_loss = running_grad_loss / len(dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], SDF Loss: {epoch_sdf_loss:.4f}, Grad Loss: {epoch_grad_loss:.4f}')

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch + 1}.pth'))

    # Save final model
    torch.save(model.state_dict(), os.path.join(model_save_path, 'model_final.pth'))
    print('Training complete and model saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SDF Model.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save the trained models')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--use_fourier', action='store_true', help='Use Fourier feature mapping')

    args = parser.parse_args()

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    train_sdf_model(args.data_dir, args.model_save_path, args.num_epochs, args.batch_size, args.learning_rate, args.use_fourier)
