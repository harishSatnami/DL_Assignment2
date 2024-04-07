## part  b
import torch
import torchvision.models as models
import torch.nn as nn


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

import wandb
import argparse

parser = argparse.ArgumentParser(description="Command line arguments for CONVOLUTIONAL NEURAL NETWORK")
parser.add_argument("-wp","--wandb_project",type=str,default="Dl_Assignment_2", help="Project name used to track experiments in Weights & Biases dashboard")
parser.add_argument("-we","--wandb_entity",type=str,default="cs23m025", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument("-trp","--train_path",type=str,default="", help="Path to training dataset")
parser.add_argument("-tsp","--test_path",type=str,default="", help="Path to testing dataset")

args = parser.parse_args()

project = args.wandb_project
entity = args.wandb_entity
train_path = args.train_path
test_path = args.test_path




if train_path=="":
    train_path = "/kaggle/input/nature-12k/inaturalist_12K/train"
if test_path=="":
    test_path = "/kaggle/input/nature-12k/inaturalist_12K/val"


def load_data(train_path, val_split=0.2, batch_size=150, data_augmentation=False):
    transform_list = [transforms.Resize((224, 224)),transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    if data_augmentation:
        transform_list.insert(0, transforms.RandomHorizontalFlip())
        transform_list.insert(0, transforms.RandomRotation(10))

    transform = transforms.Compose(transform_list)

    # Load dataset
    dataset = ImageFolder(train_path, transform=transform)
    num_classes = len(dataset.classes)
    
    # Create stratified train-validation split
    train_indices, val_indices = train_test_split(list(range(len(dataset))), 
                                                   test_size=val_split, 
                                                   stratify=dataset.targets)

    # Create DataLoader for training set
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    # Create DataLoader for validation set
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader, num_classes



def pre_trained_model_run():
    wandb.init()
    train_loader, val_loader, num_classes = load_data(train_path, val_split=0.2, batch_size=32, data_augmentation=True)
    # Load pre-trained ResNet50 model
    pretrained_resnet50 = models.resnet50(pretrained=True)

    # Freeze all convolutional layers
    for param in pretrained_resnet50.parameters():
        param.requires_grad = False

    # Replace the classifier with a new fully connected layer

    pretrained_resnet50.fc = nn.Linear(pretrained_resnet50.fc.in_features, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = optim.Adam(pretrained_resnet50.fc.parameters(), lr=learning_rate)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_resnet50.to(device)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        # Training loop
        pretrained_resnet50.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move tensors to GPU
            optimizer.zero_grad()
            outputs = pretrained_resnet50(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        # Validation loop (evaluate model performance on validation dataset)
        pretrained_resnet50.eval()
        validation_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move tensors to GPU
                outputs = pretrained_resnet50(inputs)
                val_loss = criterion(outputs, labels)
                

                validation_loss += val_loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        validation_accuracy = correct_val / total_val
        avg_val_loss = validation_loss / len(val_loader.dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, '
              f'Validation Loss: {avg_val_loss:.4f}'
              f'Validation Accuracy: {validation_accuracy:.4f}')

        run_name = "pre_trainded_{}_lr{}".format("adam", learning_rate)
        wandb.run.name = run_name
        wandb.log({"epoch": epoch+1, 
                    "train_loss": avg_train_loss, 
                    "train_accuracy": train_accuracy, 
                    "val_loss": avg_val_loss, 
                    "val_accuracy": validation_accuracy})
        
        
sweep_config = {
     "method": "bayes",
    "name" : "pre_trained_model",
    "project": project,
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"value": 0.0001},
        "dummy_param":{"values":[1,2,3,4,5]}
    }
}

wandb.login()
sweep_id = wandb.sweep(sweep_config, project=project)
wandb.agent(sweep_id, function=pre_trained_model_run,count=1)