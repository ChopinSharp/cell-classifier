import torch
import os
from torchvision import datasets, models, transforms

from PIL import Image

from transfer_learning import initialize_model, available_models_input_size


class CellClassifier:

    def __init__(self, model_file_name):
        # Load saved model
        model_dir = 'models'
        model_name, *_, dataset_mean, dataset_std = model_file_name[:-3].split('%')
        model, _ = initialize_model(model_name, num_classes=3, feature_extract=True)
        model.load_state_dict(torch.load(os.path.join(model_dir, model_file_name)))
        model.eval()
        self.classes = ['aggregated', 'normal', 'segmented']
        self.model = model
        self.input_size = available_models_input_size[model_name]
        self.data_transforms = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([float(dataset_mean)] * 3, [float(dataset_std)] * 3)
        ])

    def predict_single_img(self, url):
        img = Image.open(url).convert('RGB')
        input_tensor = self.data_transforms(img)
        input_tensor = input_tensor.view(1, *input_tensor.size())

        with torch.no_grad():
            scores = self.model(input_tensor)[0]
            pred = scores.argmax().item()
            probs = torch.nn.functional.softmax(scores, dim=0)

        print('Prediction:', self.classes[pred])
        print('Probabilities:')
        for class_name, prob in zip(self.classes, probs.tolist()):
            print('\t%s:\t%.4f' % (class_name, prob))


if __name__ == '__main__':
    classifier = CellClassifier(
        'squeezenet%2019%Mar%9%18%20%51%1.0000e-03-1.0000e-05%0.12072154879570007%0.13905109465122223.pt'
    )
    classifier.predict_single_img('data0229_dp/test/aggregated/3(2).jpg')

