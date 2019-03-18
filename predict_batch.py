from cell_classifier import CellClassifier
import os
from shutil import copy

base_dir = '/home/mwb/Downloads/3-19'
output_dir = '/home/mwb/Downloads/3-19-prediction'

os.makedirs(output_dir)

selected_model_file = \
    '/home/mwb/Documents/cell-classifier/models/squeezenet%Wed Mar 13 15#10#37 2019%0.378%0.12072%0.13905.pt'
classifier = CellClassifier(selected_model_file)

stats = {
    'hyperfused': 0,
    'WT': 0,
    'fragmented': 0
}

for folder in os.listdir(base_dir):
    input_folder = os.path.join(base_dir, folder)
    output_folder = os.path.join(output_dir, folder)
    os.makedirs(output_folder)
    folder_stats = {
        'hyperfused': 0,
        'WT': 0,
        'fragmented': 0
    }
    for idx, name in enumerate(os.listdir(input_folder)):
        url = os.path.join(input_folder, name)
        pred, prob_dict = classifier.predict_single_img(url)
        output_name = '%s-%d [ %s %.4f ].jpg' % (folder, idx, pred, prob_dict[pred])
        copy(url, os.path.join(output_folder, output_name))
        # print(url, 'done')
        stats[pred] += 1
        folder_stats[pred] += 1
    total = len(os.listdir(input_folder))
    print('%-8s:' % folder, end='')
    for k, v in folder_stats.items():
        print('     %s: %7.3f%%' % (k, 100 * v / total), end='')
    print()
    # print(folder, 'done')

print('\nTotal:', stats)
