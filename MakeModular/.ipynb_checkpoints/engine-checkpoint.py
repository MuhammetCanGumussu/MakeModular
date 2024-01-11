"""
Contains functions for training and testing a pytorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device) -> Tuple[float, float]:
    """
    Trains a pytorch model for a single epoch.

    Turns a target pytorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step)

    Args:
        model: a pytorch mdoel to be trained.
        dataloader: a dataloader object for model to be trained on
        loss_fn: a pytorch loss function to minimize
        optimizer: a pytorch optimizer to update model
        device: a target device to computed on (e.g. "cuda" or "cpu")

    Returns:
        A tuple of training loss and training accuracy metrics.
        (train_loss, train_accuracy)
    """
    # Put model in train mode
    model.train()
    
    # setup train loss and train acc values
    train_loss, train_acc = 0, 0

    # loop through dataloader batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        
        # 1- Forward pass
        y_pred = model(X)
        
        # 2- Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        # 3- Optmizer zero grad
        optimizer.zero_grad()
        
        # 4- Loss backward
        loss.backward()
        
        # 5- Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics (loss and acc) to get average loss and accuracy per batch
    train_loss = train_loss / len(data_loader) # len(dataloader) == # of batch for 1 epoch
    train_acc = train_acc / len(data_loader)
    
    return train_loss, train_acc
    
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    Tests a pytorch model for a single epoch.

    Turns a target pytorch model to "eval" mode and then performs
    a forward pass on a testing dataset via test dataloader.

    Args:
        model: a pytorch model to be tested.
        dataloader: a dataloader object for the model to be tested on
        loss_fn: a pytorch loss function to calculate loss on the test data
        device: a target device to compute on (e.g. "cuda" or "cpu")

    Returns:
        a tuple of testing loss and testing accuracy metrics.
        (test_loss, test_accuracy)
    
    """
    # change model mode to eval(evaluation)
    model.eval()
    
    # setup test loss and acc values
    test_loss, test_acc = 0, 0
   
    # turn on inference mode context manager (autograd will not track)
    with torch.inference_mode():
        
        # loop through dataloader batches
        for batch, (X, y) in enumerate(dataloader):
           
            # send data to target device
            X, y = X.to(device), y.to(device)
            
            # 1- forward pass
            test_pred_logits = model(X)
            
            # 2- calculate and accumalate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # calculate and accumulate accuracy
            test_pred_class = test_pred_logits.argmax(dim=1) 
            test_acc += (test_pred_class == y).sum().item()/len(test_pred_class)
            
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(data_loader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
         train_dataloader: torch.utils.data.DataLoader,
         test_dataloader: torch.utils.data.DataLoader,
         optimizer: torch.optim.Optimizer,
         loss_fn: torch.nn.Module,
         epochs: int,
         device: torch.device) -> Dict[str, list]:
    """
    Trains and tests a pytorch model.
    passes a target pytorch models through train_step() and
    test_step() functions for a number of epochs, training and 
    testing the model in the same epoch loop

    Calculates, prints and stores evaluation metrics throughout

    Args:
        model: a pytorch model to be trained and tested
        train_dataloader: a dataloader object for the model to be trained on
        test_dataloader: a dataloader object for the model to be tested on
        optimizer: a pytorch optimizer to help minimize the loss function
        loss_fn: a pytorch loss function to calculate loss on both datasets
        epocs: an integer indicating how many epochs to train for.
        device: a target device to compute on (e.g. "cuda" or "cpu")
    
    Returns:
        a dictionary of trainnig and testing loss as well as training
        and testing accuracy metrics. each metric has a value in a list 
        for each epoch.
        in the form of: {train_loss: [...], 
                         train_acc: [...],
                         test_loss: [...],
                         test_acc: [...]}
         e.g. train_loss: [2.003, 1.9555, 4.6] -> train loss for 3 epochs
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": [] }
    # loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # print out what's happening
        print(
            f"Epochs: {epoch+1} | "
            f"train_loss: {train_loss:.4f} |"
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
        
            
