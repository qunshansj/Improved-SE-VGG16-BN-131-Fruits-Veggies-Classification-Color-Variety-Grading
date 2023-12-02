python
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import vgg

class ImageClassifier:
    def __init__(self, model_name, num_classes, weights_path, json_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.model = vgg(model_name=model_name, num_classes=num_classes).to(self.device)
        self.weights_path = weights_path
        self.json_path = json_path

    def load_image(self, img_path):
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        img = self.data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        return img

    def load_class_indict(self):
        assert os.path.exists(self.json_path), "file: '{}' dose not exist.".format(self.json_path)
        with open(self.json_path, "r") as f:
            class_indict = json.load(f)
        return class_indict

    def load_model_weights(self):
        assert os.path.exists(self.weights_path), "file: '{}' dose not exist.".format(self.weights_path)
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))

    def predict(self, img):
        self.model.eval()
        with torch.no_grad():
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        return predict_cla, predict

    def show_result(self, class_indict, predict_cla, predict):
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        plt.title(print_res)
        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      predict[i].numpy()))
        plt.show()

    def classify_image(self, img_path):
        img = self.load_image(img_path)
        class_indict = self.load_class_indict()
        self.load_model_weights()
        predict_cla, predict = self.predict(img)
        self.show_result(class_indict, predict_cla, predict)


