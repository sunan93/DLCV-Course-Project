"""Main class, holding information about models and training/testing routines."""

import torch
from ..consts import BENCHMARK
from ..utils import cw_loss
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch

class WitchBullsEye(_Witch):
    """Brew poison frogs variant with averaged feature matching instead of sums of feature matches.

    This is also known as BullsEye Polytope Attack.

    """

    def _define_objective(self, inputs, labels, targets, intended_classes, true_classes):
        """Implement the closure here."""
        def closure(model, criterion, optimizer, target_grad, target_clean_grad, target_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            """
            if self.args.target_criterion in ['cw', 'carlini-wagner']:
                criterion = cw_loss
            else:
                pass  # use the default for untargeted or targeted cross entropy
            # Carve up the model
            feature_model, last_layer = self.bypass_last_layer(model)

            # Get standard output:
            outputs = feature_model(inputs)
            #print("shape = ", outputs.shape)
            outputs_targets = feature_model(targets)
            #print("shape = ", outputs_targets.shape)
            #prediction = (last_layer(outputs).data.argmax(dim=1) == labels).sum()
            prediction = (model(inputs).argmax(dim=1) == labels).sum()

            #print("prediction= ", outputs_targets.shape)
            feature_loss = torch.sqrt((outputs.mean(dim=0) - outputs_targets.mean(dim=0)).pow(2).mean())
            feature_loss = feature_loss / torch.sqrt(outputs_targets.mean(dim=0).pow(2).mean())
            feature_loss.backward(retain_graph=self.retain)
            return feature_loss.detach().cpu(), prediction.detach().cpu()
            """
            if self.args.target_criterion in ['cw', 'carlini-wagner']:
                criterion = cw_loss
            else:
                pass  # use the default for untargeted or targeted cross entropy
            # Carve up the model
            feature_model, last_layer, headless_model = self.bypass_last_layer(model)


            # Get standard output:
            outputs = feature_model(inputs)
            outputs_targets = feature_model(targets)
            
            outputs_2 = headless_model(inputs)
            outputs_targets_2 = headless_model(targets)
            prediction = (last_layer(outputs).data.argmax(dim=1) == labels).sum()

            feature_loss = (outputs.mean(dim=0) - outputs_targets.mean(dim=0)).pow(2).mean()
            feature_loss_2 = (outputs_2.mean(dim=0) - outputs_targets_2.mean(dim=0)).pow(2).mean()
            feature_loss += feature_loss_2
            feature_loss.backward(retain_graph=self.retain)
            return feature_loss.detach().cpu(), prediction.detach().cpu()
        return closure


    @staticmethod
    def bypass_last_layer(model):
        """Hacky way of separating features and classification head for many models.

        Patch this function if problems appear.
        """
        """
        #layer_cake = list(model.children())
        lis = []
        for module in model.children():
            lis.append(module.conv1)
            lis.append(module.bn1)
            lis.append(module.relu)
            lis.append(module.layers) 
            lis.append(module.avgpool)
        newmodel = torch.nn.Sequential(*(lis))
        #print(newmodel)
        #print(layer_cake, layer_cake[:-1])
        #last_layer = layer_cake[-1]
        #headless_model = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())  # this works most of the time all of the time :<
        #return headless_model, last_layer
        return newmodel, model 
        """
        layer_cake = list(model.children())
        last_layer = layer_cake[-1]
        headless_model_1 = torch.nn.DataParallel(torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten()))
        headless_model_2 = torch.nn.DataParallel(torch.nn.Sequential(*(layer_cake[0:2]), torch.nn.Flatten()))
        #print(headless_model_2, last_layer)
        #headless_model = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
        return headless_model_1, last_layer, headless_model_2
