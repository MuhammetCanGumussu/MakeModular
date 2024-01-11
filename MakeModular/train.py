"""
Trains a pytorch image classification model. This file is can be directly executed
via "python train.py". For making it flexiable with command line interface
I need to functionalize statements and call them with conveinent arguments with
argparse (for example: python train.py --learning_rate 0.003 --epochs 25 --batch_size 12 ...)
"""

import os
from multiprocessing import freeze_support
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms



def main(args):
    
    # setup hyperparameters
    NUM_EPOCHS = 1 if args.epoch is None else args.epoch
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001  if args.learning_rate is None else args.learning_rate
    NUM_WORKERS = os.cpu_count()
    print("Number of cpu: ", NUM_WORKERS)
    
    data_setup.download_data()
    
    # setup directories
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"
    
    # setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # create transforms
    data_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    if args.multi_processing == "True":
        print(f"multi_processing state: {args.multi_processing}")
        print("Pin memory is activated, dataloader gonna use multiple cpu's...")
        # create dataloaders with help from data_setup.py
        train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
            train_dir = train_dir,
            test_dir = test_dir,
            transform = data_transforms,
            batch_size = BATCH_SIZE,
            num_workers = NUM_WORKERS - 6,    # Dangerous code! 
            pin_memory = True
        )
    else:
        print(f"multi_processing state: {args.multi_processing}")
        print("Pin memory is not activated, dataloader gonna use one cpu...")
        # create dataloaders with help from data_setup.py
        train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
            train_dir = train_dir,
            test_dir = test_dir,
            transform = data_transforms,
            batch_size = BATCH_SIZE,
        )
    
    # create model with help from model_builder.py
    if args.model == "tiny":
        # create model with help from model_builder.py
        model = model_builder.TinyVGG(
            input_shape = 3,
            hidden_units = HIDDEN_UNITS,
            output_shape = len(class_names)
        ).to(device)
    else:
        model = model_builder.LargeVGG(
            input_shape = 3,
            hidden_units = HIDDEN_UNITS,
            output_shape = len(class_names)
        ).to(device)
        
    print(model)

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

import argparse

if __name__ == "__main__":
    freeze_support()
    #parser = argparse.ArgumentParser(description="Trains your model.")
    ## positional argument (mandatory)
    #parser.add_argument("learning_rate", type=int, help="specifies learning rate")
    ## optional argument (shorthand, longhand) (requrired true -> mandatory)
    #parser.add_argument("-e", "--epoch", type=int, help="specifies epoch number", required=True)
    #args = parser.parse_args()
    #print(args.learning_rate, args.epoch)
    
    #print(sys.argv) # when command : python train.py -> prints: ["train.py"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("--model", type=str, required=True, default="tiny")
    parser.add_argument("--multi_processing", type=str, required=True)
    args = parser.parse_args()
    
    main(args)

    
