from PIL import Image
from torchvision import transforms
from torch import Tensor
import torch

def image_loader(image_name):
    imsize = 512

    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def save_image(tensor, sava_path):
    unloader = transforms.ToPILImage()

    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(sava_path)