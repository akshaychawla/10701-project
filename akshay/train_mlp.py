from torch import nn
from caltech101_dataloader import get_loader
import torch


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)
    
    def forward(self, x):
        y = self.fc1(x) 
        y = self.act1(y) 
        y = self.fc2(y) 
        return y 

def calculate_validation_accuracy(val_loader, model, criterion):
    model.eval() 
    total_correct = 0.0 
    total_samples = 0.0
    total_loss = 0.0
    num_iterations = 0.0
    for iter_idx, (xs,ys) in enumerate(val_loader):
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

    train_loader, valid_loader = get_loader(root_folder="../101_ObjectCategories/", batch_size=128)
    hidden_size = 1000
    mlp = MLP(30000, hidden_size, num_classes=102) # including the background class  
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01)

    # train mlp 
    num_epochs = 20 
    mlp.train()
    for epoch in range(num_epochs):
        mlp.train()
        print("Epoch: {}".format(epoch))
        for iter_idx, (xs,ys) in enumerate(train_loader):
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
            print("Iteration {} | Loss: {} | Accuracy: {}".format(iter_idx, loss.item(), accuracy))

        # Validation accuracy 
        val_acc, val_loss = calculate_validation_accuracy(valid_loader, mlp, criterion)
        print("VALIDATION ACC: {} LOSS: {}".format(val_acc, val_loss))



if __name__ == "__main__":
    main()