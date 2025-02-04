import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from .loss import ContentLoss, StyleLoss
from .norm import Normalization
from torchvision import transforms


class NeuralStyleTransfer:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225], device=self.device)

    def get_style_model_and_losses(self, style_img, content_img):
        content_layers = {'conv_4'}
        style_layers = {'conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'}

        normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std).to(self.device)
        model = nn.Sequential(normalization)
        
        content_losses, style_losses = [], []
        i = 0  # Счетчик свёрточных слоев

        for layer in self.cnn.children():
            name = None
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'

            if name:
                model.add_module(name, layer.to(self.device))

                if name in content_layers:
                    target = model(content_img).detach()
                    content_loss = ContentLoss(target)
                    model.add_module(f"content_loss_{i}", content_loss)
                    content_losses.append(content_loss)

                if name in style_layers:
                    target_feature = model(style_img).detach()
                    style_loss = StyleLoss(target_feature)
                    model.add_module(f"style_loss_{i}", style_loss)
                    style_losses.append(style_loss)

        # Обрезаем модель до последнего слоя с функцией потерь
        last_layer_idx = max(i for i, layer in enumerate(model) if isinstance(layer, (ContentLoss, StyleLoss)))
        model = model[: last_layer_idx + 1]

        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):
        return optim.LBFGS([input_img.requires_grad_(True)])

    def run_style_transfer(self, content_img, style_img, input_img, num_steps=100, style_weight=1e6, content_weight=1, save_steps=True):
        model, style_losses, content_losses = self.get_style_model_and_losses(style_img, content_img)
        model.eval()

        optimizer = self.get_input_optimizer(input_img)
        intermediate_images = []
        intermediate_images.append(input_img.clone().cpu().squeeze(0))

        run = [0]
        while run[0] <= num_steps:
            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)

                style_score = sum(sl.loss for sl in style_losses) * style_weight
                content_score = sum(cl.loss for cl in content_losses) * content_weight
                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 10 == 0:
                    print('Run {} : Style Loss = {:4f} Content Loss = {:4f}'.format(
                        run[0], style_score.item(), content_score.item()))

                return loss

            optimizer.step(closure)

            if save_steps and run[0] % 5 == 0:
                with torch.no_grad():
                    img = input_img.clone().cpu().squeeze(0)
                    img = transforms.ToPILImage()(img)
                    intermediate_images.append(img)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img, intermediate_images
