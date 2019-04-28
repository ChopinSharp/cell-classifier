from scripts.train_classifier import validate_model, available_models_input_size
from visdom import Visdom


def main():
    viz = Visdom(port=2337, env='different classifiers')
    acc_list = []
    model_name_list = list(available_models_input_size.keys())
    for model_name in model_name_list:
        _, *acc = validate_model('../datasets/data0229', model_name=model_name, viz=viz, num_epochs=80,
                                 model_dir=None, script_dir=None, feature_extract=False,
                                 learning_rates=[1e-5], weight_decays=[1e-4])
        acc_list.append(acc)
    viz.bar(X=acc_list, opts=dict(rownames=model_name_list, legend=['val acc', 'test acc']))


main()