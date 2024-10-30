import functools
import operator
from lightning import seed_everything
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
import torchmetrics
import torchvision
import torchvision.models as models
from src.cindex import concordance_index
from einops import rearrange

seed_everything(2)

class Classifer(pl.LightningModule):
    def __init__(self, num_classes=9, init_lr=3e-4):
        super().__init__()
        self.init_lr = init_lr
        self.num_classes = num_classes

        # Define loss fn for classifier
        self.loss = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.auc = torchmetrics.AUROC(task="binary" if self.num_classes == 2 else "multiclass", num_classes=self.num_classes)

        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

    def get_xy(self, batch):
        if isinstance(batch, list):
            x, y = batch[0], batch[1]
        else:
            assert isinstance(batch, dict)
            x, y = batch["x"], batch["y_seq"][:,0]
        return x, y.to(torch.long).view(-1)

    def training_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        # get predictions from your model and store them as y_hat
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)

        ## Store the predictions and labels for use at the end of the epoch
        self.training_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)

        self.validation_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def test_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        self.log('test_loss', loss, sync_dist=True, prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)

        self.test_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss
    
    def on_train_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.training_outputs])
        y = torch.cat([o["y"] for o in self.training_outputs])
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)
        self.log("train_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)
        self.log("val_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

    def on_test_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.test_outputs])
        y = torch.cat([o["y"] for o in self.test_outputs])

        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)

        self.log("test_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.test_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = LinearLR(optimizer)
        return [optimizer], [scheduler]
    
    def init_weights(m, nonlinearity='relu'):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
        

class MLP(Classifer):
    def __init__(self, input_dim=28*28*3, hidden_dim=128, num_layers=1, num_classes=9, use_bn=False, init_lr=1e-3, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn
        self.bn = [nn.BatchNorm1d(hidden_dim)] if self.use_bn else []
        self.num_layers = num_layers

        self.first_layer = nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
                                         *self.bn,
                                         nn.ReLU())

        self.hidden_layers = []
        for _ in range(self.num_layers - 1):
            self.hidden_layers.append(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                    *self.bn,
                                                    nn.ReLU())
                                         )

        self.final_layer = nn.Sequential(nn.Linear(self.hidden_dim, num_classes),    
                                         nn.Softmax(dim=-1)
                                         )

        self.model = nn.Sequential(self.first_layer,
                                   *self.hidden_layers,
                                   self.final_layer
                                   )
        
        self.model.apply(self.init_weights)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = rearrange(x, 'b c w h -> b (w h c)')
        return self.model(x)


class LinearModel(Classifer):
    def __init__(self, input_dim=28*28*3, hidden_dim=128, num_layers=1, num_classes=9, use_bn=False, init_lr=1e-3, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn
        self.bn = [nn.BatchNorm1d(self.hidden_dim)] if self.use_bn else []
        self.num_layers = num_layers

        self.first_layer = nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
                                         *self.bn)

        self.hidden_layers = []
        for _ in range(self.num_layers - 1):
            self.hidden_layers.append(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                    *self.bn)
                                         )

        self.final_layer = nn.Sequential(nn.Linear(self.hidden_dim, num_classes),    
                                         nn.Softmax(dim=-1)
                                         )

        self.model = nn.Sequential(self.first_layer,
                                   *self.hidden_layers,
                                   self.final_layer
                                   )

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = rearrange(x, 'b c w h -> b (w h c)')
        return self.model(x)


class CNN(Classifer):
    def __init__(self, input_dim=(3, 28, 28), hidden_dim=128, num_layers=1, num_classes=9, use_bn=False, init_lr = 1e-3, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn
        self.bn_fc = [nn.BatchNorm1d(hidden_dim)] if self.use_bn else []
        self.num_layers = num_layers

        # initialize convolutional layers 
        self.feature_extractor = []
        for i in range(num_layers):
            if i == 0: # first conv layer
                in_channels = input_dim[0]
                out_channels = 20
                k = 5 # Conv2d kernel size
                conv_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k,k)),
                                           nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                           nn.ReLU()
                                            )
            else: # subsequent conv layers
                k = 3
                bn_conv = [nn.BatchNorm2d(out_channels)] if self.use_bn else []
                conv_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k,k)),
                                           *bn_conv,
                                           nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                           nn.ReLU()
                                           )

            self.feature_extractor.append(conv_layer)
            
            # set channels for i > 0
            in_channels = out_channels
            out_channels *= 2
        
        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        # get number of output features from conv layers
        num_features_before_fc = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape))
            
		# initialize fully connected layers
        self.classifier = []
        for i in range(num_layers):
            in_features = num_features_before_fc if i == 0 else self.hidden_dim

            if i == num_layers - 1: # add final fc layer
                fc_layer = nn.Sequential(nn.Linear(in_features, num_classes),
                                         nn.Softmax(dim=-1)
                                         )
                        
            else: # add consequent fc layres
                fc_layer = nn.Sequential(nn.Linear(in_features=in_features, out_features=self.hidden_dim),
                                        *self.bn_fc,
                                        nn.ReLU()
                                        )
        
            self.classifier.append(fc_layer)
        
        self.classifier = nn.Sequential(*self.classifier)              

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = rearrange(x, 'b c w h -> b c h w')
        x = self.feature_extractor(x).flatten(1)
        return self.classifier(x)


class ResNet18(Classifer):
    def __init__(self, num_classes=9, init_lr=1e-3, pretraining=False, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        # Initialize a ResNet18 model
        weights_kwargs = {'weights': models.ResNet18_Weights.DEFAULT} if pretraining else {} 
        backbone = models.resnet18(**weights_kwargs)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Sequential(nn.Linear(num_filters, num_classes),    
                                         nn.Softmax(dim=-1)
                                         )
        self.classifier.apply(self.init_weights)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = rearrange(x, 'b c w h -> b c h w')
        x = self.feature_extractor(x).flatten(1)
        return self.classifier(x)


NLST_CENSORING_DIST = {
    "0": 0.9851928130104401,
    "1": 0.9748317321074379,
    "2": 0.9659923988537479,
    "3": 0.9587252204657843,
    "4": 0.9523590830936284,
    "5": 0.9461840310101468,
}

class RiskModel(Classifer):
    def __init__(self, input_num_chan=1, num_classes=2, init_lr = 1e-3, max_followup=6, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = 512

        ## Maximum number of followups to predict (set to 6 for full risk prediction task)
        self.max_followup = max_followup

        # TODO: Initalize components of your model here
        raise NotImplementedError("Not implemented yet")



    def forward(self, x):
        raise NotImplementedError("Not implemented yet")

    def get_xy(self, batch):
        """
            x: (B, C, D, W, H) -  Tensor of CT volume
            y_seq: (B, T) - Tensor of cancer outcomes. a vector of [0,0,1,1,1, 1] means the patient got between years 2-3, so
            had cancer within 3 years, within 4, within 5, and within 6 years.
            y_mask: (B, T) - Tensor of mask indicating future time points are observed and not censored. For example, if y_seq = [0,0,0,0,0,0], then y_mask = [1,1,0,0,0,0], we only know that the patient did not have cancer within 2 years, but we don't know if they had cancer within 3 years or not.
            mask: (B, D, W, H) - Tensor of mask indicating which voxels are inside an annotated cancer region (1) or not (0).
                TODO: You can add more inputs here if you want to use them from the NLST dataloader.
                Hint: You may want to change the mask definition to suit your localization method

        """
        return batch['x'], batch['y_seq'][:, :self.max_followup], batch['y_mask'][:, :self.max_followup], batch['mask']

    def step(self, batch, batch_idx, stage, outputs):
        x, y_seq, y_mask, region_annotation_mask = self.get_xy(batch)

        # TODO: Get risk scores from your model
        y_hat = None ## (B, T) shape tensor of risk scores.
        # TODO: Compute your loss (with or without localization)
        loss = None

        raise NotImplementedError("Not implemented yet")
        
        # TODO: Log any metrics you want to wandb
        metric_value = -1
        metric_name = "dummy_metric"
        self.log('{}_{}'.format(stage, metric_name), metric_value, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)

        # TODO: Store the predictions and labels for use at the end of the epoch for AUC and C-Index computation.
        outputs.append({
            "y_hat": y_hat, # Logits for all risk scores
            "y_mask": y_mask, # Tensor of when the patient was observed
            "y_seq": y_seq, # Tensor of when the patient had cancer
            "y": batch["y"], # If patient has cancer within 6 years
            "time_at_event": batch["time_at_event"] # Censor time
        })

        return loss
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train", self.training_outputs)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val", self.validation_outputs)
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test", self.test_outputs)

    def on_epoch_end(self, stage, outputs):
        y_hat = F.sigmoid(torch.cat([o["y_hat"] for o in outputs]))
        y_seq = torch.cat([o["y_seq"] for o in outputs])
        y_mask = torch.cat([o["y_mask"] for o in outputs])

        for i in range(self.max_followup):
            '''
                Filter samples for either valid negative (observed followup) at time i
                or known pos within range i (including if cancer at prev time and censoring before current time)
            '''
            valid_probs = y_hat[:, i][(y_mask[:, i] == 1) | (y_seq[:,i] == 1)]
            valid_labels = y_seq[:, i][(y_mask[:, i] == 1)| (y_seq[:,i] == 1)]
            self.log("{}_{}year_auc".format(stage, i+1), self.auc(valid_probs, valid_labels.view(-1)), sync_dist=True, prog_bar=True)

        y = torch.cat([o["y"] for o in outputs])
        time_at_event = torch.cat([o["time_at_event"] for o in outputs])

        if y.sum() > 0 and self.max_followup == 6:
            c_index = concordance_index(time_at_event.cpu().numpy(), y_hat.detach().cpu().numpy(), y.cpu().numpy(), NLST_CENSORING_DIST)
        else:
            c_index = 0
        self.log("{}_c_index".format(stage), c_index, sync_dist=True, prog_bar=True)

    def on_train_epoch_end(self):
        self.on_epoch_end("train", self.training_outputs)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        self.on_epoch_end("val", self.validation_outputs)
        self.validation_outputs = []

    def on_test_epoch_end(self):
        self.on_epoch_end("test", self.test_outputs)
        self.test_outputs = []
