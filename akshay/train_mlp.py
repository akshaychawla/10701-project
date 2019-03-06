from torch import nn
from caltech101_dataloader import get_loader
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os, sys


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)
    
    def forward(self, x):
        y = self.fc1(x) 
        y = self.act1(y) 
        y = self.fc2(y) 
        return y 

def calculate_metrics(loader, model, criterion):
    model.eval() 
    total_correct = 0.0 
    total_samples = 0.0
    total_loss = 0.0
    num_iterations = 0.0
    for iter_idx, (xs,ys) in enumerate(loader):
        xs = xs.cuda() 
        ys = ys.cuda()
        output = model(xs)
        ys_preds = torch.argmax(output, 1)
        loss = criterion(output, ys)
        correct = (ys_preds==ys).sum().item() 
        total_loss += loss.item() 
        total_correct += correct 
        total_samples += len(ys) 
        num_iterations += 1.0
    
    accuracy = total_correct / total_samples 
    avg_loss = total_loss / num_iterations

    return accuracy, avg_loss

    
def main():

    train_loader, valid_loader = get_loader(root_folder="../101_ObjectCategories/", batch_size=128, num_workers=5, pin_memory=True)
    hidden_size = int(sys.argv[1])
    mlp = MLP(30000, hidden_size, num_classes=102) # including the background class  
    mlp.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # log 
    writer = SummaryWriter(comment="num_hidden = {}".format(hidden_size))


    # train mlp 
    num_epochs = 100 
    mlp.train()
    for epoch in range(num_epochs):
        mlp.train()
        print("Epoch: {}".format(epoch))
        for iter_idx, (xs,ys) in tqdm(enumerate(train_loader)):
            xs = xs.cuda() 
            ys = ys.cuda()
            output = mlp(xs)
            ys_preds = torch.argmax(output, 1)
            loss = criterion(output, ys) 

            # Calculate batch accuracy
            correct = (ys_preds == ys).sum().item()
            accuracy = correct / len(ys)

            # backprop + step 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            # print("Iteration {} | Loss: {} | Accuracy: {}".format(iter_idx, loss.item(), accuracy))

        # Dataset wide metrics 
        train_acc, train_loss = calculate_metrics(train_loader, mlp, criterion)
        val_acc, val_loss = calculate_metrics(valid_loader, mlp, criterion)
        print("TRAIN ACC: {} LOSS: {} | VALIDATION ACC: {} LOSS: {}".format(train_acc, train_loss, val_acc, val_loss))
        scheduler.step(val_loss)

        # log 
        writer.add_scalar("train/loss", train_loss, epoch+1)
        writer.add_scalar("train/acc", train_acc, epoch+1)
        writer.add_scalar("val/loss", val_loss, epoch+1)
        writer.add_scalar("val/acc", val_acc, epoch+1)



if __name__ == "__main__":
    main()