import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import argparse
import wandb


parser = argparse.ArgumentParser(description="Command line arguments for CONVOLUTIONAL NEURAL NETWORK")

parser.add_argument("-wp","--wandb_project",type=str,default="Dl_Assignment_2", help="Project name used to track experiments in Weights & Biases dashboard")
parser.add_argument("-we","--wandb_entity",type=str,default="cs23m025", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument("-trp","--train_path",type=str,default="", help="Path to training dataset")
parser.add_argument("-tsp","--test_path",type=str,default="", help="Path to testing dataset")
parser.add_argument("-e","--epochs",type=int,default=5, help="Number of epochs to train neural network.")
parser.add_argument("-b","--batch_size",default=64,type=int, help="Batch size used to train neural network.")

parser.add_argument("-lr","--learning_rate",default=0.001,type=float, help="Learning rate used to optimize model parameters")

parser.add_argument("-ld","--lr_decay",default=0.0001,type=float, help="lr decay used by optimizers.")

parser.add_argument("-a","--activation",default="mish",type=str, choices= ['mish', 'relu', 'silu', 'gelu'])

parser.add_argument("-bn","--batch_norm",default=True,type=bool, choices= [False, True])
parser.add_argument("-da","--data_augmentation",default=False,type=bool, choices= [False, True])

parser.add_argument("-d","--dropout",default=0.2,type=int, help="dropout value")
parser.add_argument("-nf","--num_of_filters",default=128,type=int, help="number of filters in the network")
parser.add_argument("-fs","--filter_size",default=5,type=int, help="size of filters")
parser.add_argument("-dn","--dense_unit",default=512,type=int, help="Dense layer ")
parser.add_argument("-fo","--filter_organization",default="double",type=str, choices=["double", "same", "halve"])

args = parser.parse_args()

project = args.wandb_project
entity = args.wandb_entity
train_path = args.train_path
test_path = args.test_path
epochs = args.epochs
batch_size = args.batch_size 

learning_rate = args.learning_rate
lr_decay = args.lr_decay
activation_conv = args.activation
dropout = args.dropout
num_of_filters = args.num_of_filters 
filter_size = args.filter_size  
dense_unit = args.dense_unit   
data_aug = args.data_augmentation 
batch_norm = args.batch_norm
filter_organization = args.filter_organization

sweep_config = {
    "method": "bayes",
    "name" : "CNN",
    "project": project,
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "num_filters": {"value": num_of_filters},
        "filter_size": {"value": filter_size},
        "dense_units": {"value": dense_unit},
        "activation_conv": {"value": activation_conv},
        "batch_size": {"value": batch_size},
        "learning_rate": {"value": learning_rate},
        "lr_decay": {"value": lr_decay},
        "num_epochs": {"value": epochs},
        "data_augmentation": {"value": data_aug},
        "batch_norm": {"value": batch_norm},
        "dropout": {"value": dropout},
        "filter_organization": {"value": filter_organization},
        "dummy_param":{ "values": [1,2,3,4,5,6]}
    }
}  
     
if train_path=="":
    train_path = "/kaggle/input/nature-12k/inaturalist_12K/train"
if test_path=="":
    test_path = "/kaggle/input/nature-12k/inaturalist_12K/val"


def load_data(train_path, val_split=0.2, batch_size=150, data_augmentation=False):
    transform_list = [transforms.Resize((224, 224)),transforms.ToTensor()]
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



class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters, filter_size, dense_units, activation_conv='relu', data_augmentation=False, batch_norm=False, dropout=0.0, filter_organization='same'):
        super(CNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(5):
            if filter_organization == 'same':
                current_num_filters = num_filters
            elif filter_organization == 'double':
                current_num_filters = num_filters * (2 ** i)
            elif filter_organization == 'halve':
                current_num_filters = num_filters // (2 ** i)

            self.conv_layers.append(nn.Conv2d(input_channels, current_num_filters, filter_size, padding=1))
            if batch_norm:
                self.conv_layers.append(nn.BatchNorm2d(current_num_filters))
            if activation_conv == 'relu':
                self.conv_layers.append(nn.ReLU(inplace=True))
            elif activation_conv == 'gelu':
                self.conv_layers.append(nn.GELU())
            elif activation_conv == 'silu':
                self.conv_layers.append(nn.SiLU())
            elif activation_conv == 'mish':
                self.conv_layers.append(nn.Mish())
            self.conv_layers.append(nn.Dropout2d(dropout))
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2))
            input_channels = current_num_filters

        input_size = 224  
        for k in range(5):
            
            input_size = (input_size - filter_size +3)
            input_size = (input_size - 2)//2 + 1
            
        self.fc = nn.Linear(input_channels * input_size * input_size, dense_units)
        self.out = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = torch.flatten(x, 1)
        x = torch.relu(self.fc(x))
        x = self.out(x)
        return x
    



def validate_model(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            val_loss += loss.item() * imgs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()
    
    # Calculate average loss and accuracy
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = 100 * correct / total
    
    return avg_val_loss, val_accuracy


def train_model():
    wandb.init()
    config = wandb.config

    # Load train data
    train_loader, val_loader, num_classes = load_data(train_path, val_split=0.2, batch_size=config.batch_size, data_augmentation=config.data_augmentation)

    # Define the CNN model
    model = CNN(input_channels=3, num_classes=num_classes, num_filters=config.num_filters,
                filter_size=config.filter_size, dense_units=config.dense_units, activation_conv=config.activation_conv,
                data_augmentation=config.data_augmentation, batch_norm=config.batch_norm, dropout=config.dropout,
                filter_organization=config.filter_organization)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=config.lr_decay)

    # Train the above model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += lbls.size(0)
            correct_train += (predicted == lbls).sum().item()
        
        # Calculate training loss and accuracy
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train

        # Validation
        avg_val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        # Log metrics to wandb
        run_name = "{}_lr{}_batchsize{}_filter{}_dense{}_decay{}_dataAug{}_batchnorm{}_dropout{}_filterorg{}".format(config.activation_conv, config.learning_rate, config.batch_size, config.num_filters, config.dense_units, config.lr_decay,
                            config.data_augmentation, config.batch_norm,config.dropout,config.filter_organization)
        wandb.run.name = run_name
        wandb.log({"epoch": epoch+1, 
                   "train_loss": avg_train_loss, 
                   "train_accuracy": train_accuracy, 
                   "val_loss": avg_val_loss, 
                   "val_accuracy": val_accuracy})
        
        print("Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Loss: {:.4f}, Val Acc: {:.2f}%".format(
            epoch+1, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy))
        
        # Adjust learning rate
        scheduler.step()
        


wandb.login()
        
sweep_id = wandb.sweep(sweep_config, project=project)
wandb.agent(sweep_id, function=train_model,count=1)