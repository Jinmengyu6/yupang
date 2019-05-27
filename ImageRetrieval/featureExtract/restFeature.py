"""ImageNet Pretrained Models"""
import torch
from PIL import Image
from torchvision import models
from torchvision import transforms
import time


class DeepFeat():

  def __init__(self):
    self.Model = models.resnet18(pretrained=True)
    self.Model.eval()

  def preprocess_for_eval(self, image):
    """ Preprocesses the given image for evaluation. """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])(image.convert('RGB'))

  def __call__(self, path):
    with Image.open(path) as img:
      img = self.preprocess_for_eval(img)[None, ]
      with torch.no_grad():
        t1 = time.time()
        outputs = self.Model.forward(img.type(torch.FloatTensor))
        # conf = torch.topk(torch.nn.functional.softmax(outputs), 1)[0][0].item()
        # t2 = time.time()
        # print((t2-t1)*1000)
        return outputs

  @staticmethod
  def match(feature1, feature2):
    # 计算欧式距离
    return torch.pow(feature1 - feature2, 2).sum().item()


if __name__ == "__main__":
  deep = DeepFeat()
  img1 = deep('1.png')
  img2 = deep('2.png')
  img3 = deep('3.png')
