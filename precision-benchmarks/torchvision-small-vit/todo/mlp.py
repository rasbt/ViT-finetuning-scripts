import time

import lightning as L
from lightning import Fabric
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import torchmetrics
from torchvision import transforms
from torchvision import datasets
from lightning.fabric.plugins import BitsandbytesPrecision
from torch.utils.data import DataLoader


# Regular PyTorch Module
class PyTorchModel(torch.nn.Module):
    def __init__(self, input_size, hidden_units, num_classes):
        super().__init__()

        # Initialize MLP layers
        all_layers = []
        for hidden_unit in hidden_units:
            layer = torch.nn.Linear(input_size, hidden_unit, bias=False)
            all_layers.append(layer)
            all_layers.append(torch.nn.ReLU())
            input_size = hidden_unit

        output_layer = torch.nn.Linear(
            in_features=hidden_units[-1],
            out_features=num_classes)

        all_layers.append(output_layer)
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # to make it work for image inputs
        x = self.layers(x)
        return x  # x are the model's logits


def train(num_epochs, model, optimizer, train_loader, val_loader, fabric, scheduler):

    for epoch in range(num_epochs):
        train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(fabric.device)

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            model.train()

            ### FORWARD AND BACK PROP
            logits = model(features)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            fabric.backward(loss)
            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            ### LOGGING
            if not batch_idx % 300:
                fabric.print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {loss:.4f}")

            model.eval()
            with torch.no_grad():
                predicted_labels = torch.argmax(logits, 1)
                train_acc.update(predicted_labels, targets)
        scheduler.step()

        ### MORE LOGGING
        model.eval()
        with torch.no_grad():
            val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(fabric.device)

            for (features, targets) in val_loader:
                outputs = model(features)
                predicted_labels = torch.argmax(outputs, 1)
                val_acc.update(predicted_labels, targets)

            fabric.print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Train acc.: {train_acc.compute()*100:.2f}% | Val acc.: {val_acc.compute()*100:.2f}%")
            train_acc.reset(), val_acc.reset()


if __name__ == "__main__":

    print("PyTorch:", torch.__version__)
    print("Lightning:", L.__version__)
    torch.set_float32_matmul_precision("medium")

    L.seed_everything(123)

    train_dataset = datasets.MNIST(root='data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = datasets.MNIST(root='data',
                                train=False,
                                transform=transforms.ToTensor())


    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=64,
                            shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=64,
                            shuffle=False)

    #########################################
    ### 2 Initializing the Model
    #########################################

    model = PyTorchModel(
        input_size=28*28,
        hidden_units=[500, 500],
        num_classes=10
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    #########################################
    ### 3 Launch Fabric
    #########################################

    # this will also use `bfloat16` by default
    precision = BitsandbytesPrecision("nf4")
    fabric = Fabric(plugins=precision)
    fabric.launch()

    train_loader, val_loader, test_loader = fabric.setup_dataloaders(
        train_loader, train_loader, test_loader)
    model, optimizer = fabric.setup(model, optimizer)

    #########################################
    ### 4 Finetuning
    #########################################

    start = time.time()
    train(
        num_epochs=3,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        fabric=fabric,
        scheduler=scheduler
    )

    end = time.time()
    elapsed = end-start
    fabric.print(f"Time elapsed {elapsed/60:.2f} min")
    fabric.print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    #########################################
    ### 5 Evaluation
    #########################################

    with torch.no_grad():
        model.eval()
        test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(fabric.device)

        for (features, targets) in test_loader:
            outputs = model(features)
            predicted_labels = torch.argmax(outputs, 1)
            test_acc.update(predicted_labels, targets)

    fabric.print(f"Test accuracy {test_acc.compute()*100:.2f}%")