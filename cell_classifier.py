import torch
import os
from torchvision import transforms
from PIL import Image
from transfer_learning import available_models_input_size, train


class CellClassifier:

    def __init__(self, model_file_url):
        # Load saved model
        model_file_name = os.path.split(model_file_url)[1]
        model_name, _, temperature, dataset_mean, dataset_std = model_file_name[:-3].split('%')
        script_model = torch.jit.load(model_file_url)
        self.classes = ['hyperfused', 'WT', 'fragmented']
        self.model = script_model
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

        prob_dict = {class_name: prob for class_name, prob in zip(self.classes, probs.tolist())}

        return self.classes[pred], prob_dict


def format_as_table(model_files):
    info_table_str = ' ID  | Model Type | Temperature | Timestamp\n'
    for idx, model_file in enumerate(model_files, 1):
        info_list = model_file.split('%')[:3]
        info_table_str += \
            ' [%d] | %-10s | %-11.3f | %s\n' % (idx, info_list[0], float(info_list[2]), info_list[1].replace('#', ':'))

    return info_table_str


def run_shell():
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)

    print('************************************************************')
    print('*                     CELL CLASSIFIER                      *')
    print('*                       version 0.2                        *')
    print('************************************************************')

    while True:
        print('====================== Main Menu ===========================')
        print('[1] Train Model with Data')
        print('[2] Load Model and Infer')
        print('[q] Quit Application')
        print('============================================================\n')

        choice = input('Your Choice: ').strip()
        while choice not in {'1', '2', 'q'}:
            print('Invalid Choice ...')
            choice = input('Your Choice: ').strip()
        print()

        # Quit
        if choice == 'q':
            break

        # Train
        elif choice == '1':
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
            if not model_files:
                print('No model files found, train a model first')
            else:
                print('Available model files:')
                print(format_as_table(model_files))
                idx = int(input('Select model to use: ')) - 1
                selected_model_file = os.path.join(model_dir, model_files[idx])
                classifier = CellClassifier(selected_model_file)
                input_file = input('Input File URL: ')
                pred, prob_dict = classifier.predict_single_img(input_file)
                print('Prediction:', pred)
                for class_name, prob in prob_dict.items():
                    print('\t%s:\t%.4f' % (class_name, prob))
            input('[ Press Enter to Return to Main Menu ]\n')


if __name__ == '__main__':
    run_shell()

