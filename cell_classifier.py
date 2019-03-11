import torch
import os
from torchvision import transforms
from PIL import Image
from transfer_learning import initialize_model, available_models_input_size, train


class CellClassifier:

    def __init__(self, model_file_url):
        # Load saved model
        model_file_name = os.path.split(model_file_url)[1]
        model_name, *_, temperature, dataset_mean, dataset_std = model_file_name[:-3].split('%')
        model, _ = initialize_model(model_name, num_classes=3, feature_extract=True)
        model.load_state_dict(torch.load(model_file_url))
        model.eval()
        self.classes = ['aggregated', 'normal', 'segmented']
        self.model = model
        self.temperature = float(temperature)
        self.input_size = available_models_input_size[model_name]
        self.data_transforms = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([float(dataset_mean)] * 3, [float(dataset_std)] * 3)
        ])

    def predict_single_img(self, url):
        img = Image.open(url).convert('RGB')
        input_tensor = self.data_transforms(img).unsqueeze(dim=0)

        with torch.no_grad():
            scores = self.model(input_tensor)[0]
            pred = scores.argmax().item()
            probs = torch.nn.functional.softmax(scores.div(self.temperature), dim=0)

        print('Prediction:', self.classes[pred])
        print('Probabilities:')
        for class_name, prob in zip(self.classes, probs.tolist()):
            print('\t%s:\t%.4f' % (class_name, prob))


if __name__ == '__main__':
    model_dir = '../models'
    print("""
************************************************************
*                     CELL CLASSIFIER                      *
*                       version 0.1                        *
************************************************************
""")

    while True:
        print('====================== Main Menu ===========================')
        print('[1] Train Model with Data')
        print('[2] Load Model and Infer')
        print('[q] Quit Application')
        print('============================================================\n')

        choice = input('Your Choice: ').strip()
        if choice == 'q':
            break
        while choice not in {'1', '2'}:
            print('Invalid Choice ...')
            choice = input('Your Choice: ').strip()
        print()

        # Train
        if choice == '1':
            print('======================= Training ===========================\n')
            data_dir = input('Training Data Folder: ')
            num_epoch = int(input('Number of epoch: '))
            print('\nStart Training ...\n')
            train(data_dir, num_epochs=num_epoch, model_dir=model_dir)
            input('[ Press Enter to Return to Main Menu ]\n')
        # Infer
        else:
            print('======================= Inferring ==========================\n')
            model_files = os.listdir(model_dir)
            print('Available model files:')
            for idx, model_file in enumerate(model_files, 1):
                print('[%d] %s' % (idx, model_file))
            idx = int(input('Select model to use: ')) - 1
            selected_model_file = os.path.join(model_dir, model_files[idx])
            classifier = CellClassifier(selected_model_file)

            input_file = input('Input File URL: ')
            classifier.predict_single_img(input_file)
            input('[ Press Enter to Return to Main Menu ]\n')

