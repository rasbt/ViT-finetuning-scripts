import lightning as L
import torch
from torchvision.models import vit_l_16
from torchvision.models import ViT_L_16_Weights
from torchvision import transforms

from shared_utilities import get_dataloaders_cifar10
from trainer_utilities import LightningModel

####################
# Initialize Model
####################

L.seed_everything(123)
pytorch_model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)

# replace output layer
pytorch_model.heads.head = torch.nn.Linear(in_features=1024, out_features=10)


####################
# Load Dataset
####################

train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       # transforms.RandomCrop((224, 224)),
                                       transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      # transforms.CenterCrop((224, 224)),
                                      transforms.ToTensor()])
train_loader, val_loader, test_loader = get_dataloaders_cifar10(
    batch_size=32,
    num_workers=4,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    validation_fraction=0.1,
    download=True
)

####################
# Train Model
####################

lightning_model = LightningModel(model=pytorch_model, learning_rate=5e-5)

trainer = L.Trainer(
    max_epochs=3,
    accelerator="gpu",
    devices=1,
    precision="bf16-true",
    deterministic=True,
)

trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model=lightning_model, dataloaders=test_loader)