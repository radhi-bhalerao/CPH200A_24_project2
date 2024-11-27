import functools
import operator
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
import torchmetrics
import torchvision.models as models
from src.cindex import concordance_index
from einops import rearrange
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
import os
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_curve
from NLST_data_dict import subgroup_dict
import warnings
from sklearn.exceptions import UndefinedMetricWarning

dirname = os.path.dirname(__file__)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

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

        if self.trainer.datamodule.name == 'NLST':
            self.validation_outputs[-1].update({
                            "criteria": batch[self.trainer.datamodule.criteria],
                            **{k:batch[k] for k in self.trainer.datamodule.group_keys},
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

        if self.trainer.datamodule.name == 'NLST':
            self.validation_outputs[-1].update({
                            "criteria": batch[self.trainer.datamodule.criteria],
                            **{k:batch[k] for k in self.trainer.datamodule.group_keys},
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

    def on_validation_start(self):
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

    def on_save_checkpoint(self, checkpoint):
        self.roc_analysis_across_nodes(self.validation_outputs)        
        return super().on_save_checkpoint(checkpoint)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = LinearLR(optimizer)
        return [optimizer], [scheduler]
    
    def init_weights(m, nonlinearity='relu'):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)

    def roc_analysis_across_nodes(self, split_outputs):
        if self.trainer.datamodule.name == 'NLST':
            # collect outputs across samples into a single tensor
            output_across_samples = {}
            for k in ['y', 'y_hat', 'criteria', *self.trainer.datamodule.group_keys]:
                output = torch.cat([o[k] for o in split_outputs])
                if k == 'y_hat':
                    if self.num_classes == 2:
                        output = F.softmax(output, dim=-1)[:,-1]
                    else:
                        output = F.softmax(output, dim=-1)
                output = output.view(-1)
                output_across_samples.update({k: output})
            
            # init empty tensors to gather tensors across notes
            output_across_nodes = {}
            for k,v in output_across_samples.items():
                output_across_nodes[k] = torch.zeros((int(self.trainer.world_size * len(v)))).type(v.type()).to(self.device).contiguous()

            # wait for all nodes to catch up
            torch.distributed.barrier()

            # gather each tensor necessary for ROC plotting
            for name, tensor in output_across_samples.items():
                torch.distributed.all_gather_into_tensor(output_across_nodes[name], tensor.contiguous(), async_op=False)

            if self.global_rank == 0: # on node 0
                # send to numpy
                output_across_nodes = {k:self.safely_to_numpy(v) for k,v in output_across_nodes.items()}

                # transform arrays as needed
                for k in self.trainer.datamodule.group_keys:
                    if k in self.trainer.datamodule.vectorizer.features_fit:
                        output_across_nodes.update(self.trainer.datamodule.vectorizer.transform({k: output_across_nodes[k]}))
                    
                # print(probs, y.view(-1), group_data) # samples are 2x batch_size
                self.roc_analysis(y=output_across_nodes['y'], 
                                    y_hat=output_across_nodes['y_hat'],
                                    criteria=output_across_nodes['criteria'],
                                    group_data={k:v for k,v in output_across_nodes.items() if k in self.trainer.datamodule.group_keys},
                                    plot_label='val set'
                                    )

    @ staticmethod
    def safely_to_numpy(tensor):
        return tensor.to(torch.float).cpu().numpy().squeeze()

    @staticmethod
    def plot_roc_operation_point(y, y_hat, ax, plot_label):
        fpr, tpr, _ = roc_curve(y, y_hat)
        ax.plot(fpr[1], tpr[1], 'go', label=f'{plot_label} operation point')
    
    def roc_analysis(self, y, y_hat, criteria, plot_label, group_data=None):
        # generate plots
        roc_plot = self.generate_roc(y, y_hat, criteria)
        print('ROC curve generated.')
        if group_data:
            subgroup_roc_plot = self.generate_subgroup_roc(y, y_hat, criteria, group_data)
            print('Subgroup ROC curve generated.')
        
        # generate plot names
        plot_name = f'ROC, {plot_label}, epoch {self.current_epoch}'
        subgroup_plot_name = f'ROC by subgroups, {plot_label}, epoch {self.current_epoch}'
        
        # log plots
        if self.logger.experiment:
            wandb_logger = self.logger
            wandb_logger.log_image(key=plot_name, images=[roc_plot])
            if group_data:
                wandb_logger.log_image(key=subgroup_plot_name, images=[subgroup_roc_plot])


    def generate_roc(self, y, y_hat, criteria):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))

        RocCurveDisplay.from_predictions(y,
                                        y_hat,
                                        name=None,
                                        plot_chance_level=True,
                                        ax=ax)
                    
        self.plot_roc_operation_point(y,
                                      criteria,
                                      ax=ax,
                                      plot_label=self.trainer.datamodule.criteria)

        ax.legend(loc='lower right', prop={'size': 8})

        return fig

    def generate_subgroup_roc(self, y, y_hat, criteria, group_data):
        # set up figure
        ncols = 2
        nrows = ceil(len(group_data)/ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(int(ncols*10), (int(nrows*8))))
        axs = axs.ravel()
        axs_count = 0

        for group_key, group_i in group_data.items():
            subgroups = np.unique(group_i)

            for j, subgroup in enumerate(subgroups):
                # get group indices
                subgroup_idxs = np.argwhere(group_i == subgroup)

                # get subgroup name
                if group_key in self.trainer.datamodule.vectorizer.features_fit:
                    subgroup_name = self.trainer.datamodule.vectorizer.feature_levels[group_key][j]
                else:
                    subgroup_name = subgroup_dict[group_key][subgroup]                    

                # get roc curve kwargs
                roc_kwargs = dict(name=subgroup_name, 
                                  pos_label=1,
                                  ax=axs[axs_count])
                
                if j == len(subgroups) - 1:
                    roc_kwargs.update(dict(plot_chance_level=True))

                # generate roc curve
                RocCurveDisplay.from_predictions(y[subgroup_idxs],
                                                 y_hat[subgroup_idxs], 
                                                 **roc_kwargs)
            
            # Plot operation point for_criteria
            self.plot_roc_operation_point(y[subgroup_idxs],
                                          criteria[subgroup_idxs],
                                          ax=axs[axs_count],
                                          plot_label=self.trainer.datamodule.criteria)

            axs[axs_count].grid()
            axs[axs_count].legend(loc='lower right', prop={'size': 8})
            axs_count += 1
        
        # remove unused subplots
        for i in range(axs_count, int(nrows*ncols)):
            fig.delaxes(axs[i])

        plt.tight_layout()

        return fig        

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


    def __init__(self, input_dim=(3, 16, 28, 28), hidden_dim=128, num_layers=1, num_classes=9, use_bn=False, init_lr=1e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn
        self.bn_fc = [nn.BatchNorm1d(hidden_dim)] if self.use_bn else []
        self.num_layers = num_layers
        self.init_lr = init_lr

        # Initialize convolutional layers
        self.feature_extractor = []
        for i in range(num_layers):
            if i == 0:  # First conv layer
                in_channels = input_dim[0]
                out_channels = 20
                k = 3  # Conv3d kernel size
                conv_layer = nn.Sequential(
                    nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k, k, k)),
                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                    nn.ReLU()
                )
            else:  # Subsequent conv layers
                k = 3
                bn_conv = [nn.BatchNorm3d(out_channels)] if self.use_bn else []
                conv_layer = nn.Sequential(
                    nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k, k, k)),
                    *bn_conv,
                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                    nn.ReLU()
                )

            self.feature_extractor.append(conv_layer)

            # Update channels for layers i > 0
            in_channels = out_channels
            out_channels *= 2

        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        # Get the number of output features from conv layers
        num_features_before_fc = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape[1:]))

        # Initialize fully connected layers
        self.classifier = []
        for i in range(num_layers):
            in_features = num_features_before_fc if i == 0 else self.hidden_dim

            if i == num_layers - 1:  # Final fc layer
                fc_layer = nn.Sequential(
                    nn.Linear(in_features, num_classes),
                    nn.Softmax(dim=-1)
                )
            else:  # Hidden fc layers
                fc_layer = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=self.hidden_dim),
                    *self.bn_fc,
                    nn.ReLU()
                )

            self.classifier.append(fc_layer)

        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        # x shape: (batch_size, channels, depth, height, width)
        x = self.feature_extractor(x).flatten(1)
        return self.classifier(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
class CNN3D(Classifer):
    def __init__(self, input_dim=(3, 16, 28, 28), hidden_dim=128, num_layers=1, num_classes=9, use_bn=False, init_lr=1e-3, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn
        self.bn_fc = [nn.BatchNorm1d(hidden_dim)] if self.use_bn else []
        self.num_layers = num_layers

        # Initialize convolutional layers
        self.feature_extractor = []
        for i in range(num_layers):
            if i == 0:  # First conv layer
                in_channels = input_dim[0]
                out_channels = 20
                k = 3  # Conv3d kernel size
                conv_layer = nn.Sequential(
                    nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k, k, k)),
                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                    nn.ReLU()
                )
            else:  # Subsequent conv layers
                k = 3
                bn_conv = [nn.BatchNorm3d(out_channels)] if self.use_bn else []
                conv_layer = nn.Sequential(
                    nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k, k, k)),
                    *bn_conv,
                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                    nn.ReLU()
                )

            self.feature_extractor.append(conv_layer)

            # Update channels for layers i > 0
            in_channels = out_channels
            out_channels *= 2

        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        # Get the number of output features from conv layers
        num_features_before_fc = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape[1:]))

        # Initialize fully connected layers
        self.classifier = []
        for i in range(num_layers):
            in_features = num_features_before_fc if i == 0 else self.hidden_dim

            if i == num_layers - 1:  # Final fc layer
                fc_layer = nn.Sequential(
                    nn.Linear(in_features, num_classes),
                    nn.Softmax(dim=-1)
                )
            else:  # Hidden fc layers
                fc_layer = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=self.hidden_dim),
                    *self.bn_fc,
                    nn.ReLU()
                )

            self.classifier.append(fc_layer)

        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        # x shape: (batch_size, channels, depth, height, width)
        x = self.feature_extractor(x).flatten(1)
        return self.classifier(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
class ResNet18(Classifer):
    def __init__(self, num_classes=9, init_lr=1e-3, pretraining=False, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        # Initialize a ResNet18 model
        weights_kwargs = {'weights': models.ResNet18_Weights.DEFAULT} if pretraining else {} 
        self.classifier = models.resnet18(**weights_kwargs)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

        if not pretraining:
            self.classifier.apply(self.init_weights)

    def forward(self, x):
        print('Size: ', x.size())
        batch_size, channels, width, height = x.size()
        x = rearrange(x, 'b c w h -> b c h w')
        return self.classifier(x)

class ResNet18_adapted(Classifer):
    def __init__(self, num_classes=9, init_lr=1e-3, pretraining=False, depth_handling='max_pool', **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()
        
        # Initialize a ResNet18 model
        weights_kwargs = {'weights': models.ResNet18_Weights.DEFAULT} if pretraining else {} 
        self.classifier = models.resnet18(**weights_kwargs)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        
        # Add handling for depth dimension
        self.depth_handling = depth_handling
        
        if depth_handling == '3d_conv':
            # Replace first conv layer with 3D conv
            self.classifier.conv1 = nn.Conv3d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False
            )
            # Add a transition layer after first conv to go back to 2D
            self.transition = nn.Sequential(
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
            )
        
        if not pretraining:
            self.classifier.apply(self.init_weights)

    def forward(self, x):
        # x shape: [batch_size, channels, depth, height, width]
        batch_size, channels, depth, height, width = x.size()
        
        if self.depth_handling == 'max_pool':
            # Method 1: Max pool across depth dimension
            x = x.max(dim=2)[0]  # Shape becomes [batch_size, channels, height, width]
            return self.classifier(x)
            
        elif self.depth_handling == 'avg_pool':
            # Method 2: Average pool across depth dimension
            x = x.mean(dim=2)  # Shape becomes [batch_size, channels, height, width]
            return self.classifier(x)
            
        elif self.depth_handling == 'slice_attention':
            # Method 3: Learn attention weights for each slice
            # First, reshape to treat depth as batch dimension
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(-1, channels, height, width)
            
            # Get features for each slice
            features = self.classifier.conv1(x)
            features = self.classifier.bn1(features)
            features = self.classifier.relu(features)
            features = self.classifier.maxpool(features)
            features = self.classifier.layer1(features)
            
            # Reshape back to include depth
            features = features.view(batch_size, depth, -1)
            
            # Simple attention mechanism
            attention_weights = F.softmax(self.classifier.avgpool(features), dim=1)
            weighted_features = (features * attention_weights).sum(dim=1)
            
            # Continue through rest of network
            x = self.classifier.layer2(weighted_features)
            x = self.classifier.layer3(x)
            x = self.classifier.layer4(x)
            x = self.classifier.avgpool(x)
            x = torch.flatten(x, 1)
            return self.classifier.fc(x)
            
        elif self.depth_handling == '3d_conv':
            # Method 4: Start with 3D convolutions
            x = self.classifier.conv1(x)
            x = self.transition(x)
            
            # Reshape to 2D after initial 3D conv
            x = x.squeeze(2)  # Remove depth dimension
            
            # Continue through rest of the network
            x = self.classifier.bn1(x)
            x = self.classifier.relu(x)
            x = self.classifier.maxpool(x)
            x = self.classifier.layer1(x)
            x = self.classifier.layer2(x)
            x = self.classifier.layer3(x)
            x = self.classifier.layer4(x)
            x = self.classifier.avgpool(x)
            x = torch.flatten(x, 1)
            return self.classifier.fc(x)

class ResNet3D(Classifer):
    def __init__(self, num_classes=2, init_lr=1e-3, pretraining=False, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        if pretraining:
            backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        else:
            backbone = r3d_18(weights=None)
        
        # Modify the classifier
        num_features = backbone.fc.in_features
        backbone.fc = nn.Linear(num_features, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)

class Swin3DModel(Classifer):
    def __init__(self, num_classes=2, init_lr=1e-3, pretraining=True, num_channels=3, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        if pretraining:
            weights = Swin3D_B_Weights.DEFAULT
            self.model = swin3d_b(weights=weights)
        else:
            self.model = swin3d_b(weights=None)

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


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
