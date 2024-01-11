"""
Trains a pytorch image classification model. This file is can be directly executed
via "python train.py". For making it flexiable with command line interface
I need to functionalize statements and call them with conveinent arguments with
argparse (for example: python train.py --learning_rate 0.003 --epochs 25 --batch_size 12 ...)
"""

import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE =32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# setup directories
train_dir = "data/pizza_steak_sushi_train"
test_dir = "data/pizza_steak_sushi/test"

# setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# create transforms
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# create dataloaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir = train_dir,
    test_dir = test_dir,
    transform = data_transforms,
    batch_size = BATCH_SIZE
)

# create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape = 3,
    hidden_units = HIDDEN_UNITS,
    output_shape = len(class_names)
).to(device)

# set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# start training with help from engine.py
engine.train(model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=NUM_EPOCHS,
            device=device)

# save the model with help from utils.py
utils.save_model(model=model,
                target_dir="models",
                model_name="MakeModular_tinyvgg_modelV0.pth")
    
