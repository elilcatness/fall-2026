import numpy as np
import pandas as pd

from src.model import Model
from src.dataset import DataSet
from src.linear_classificator import LinearClassificator
from src.plot import plot_margins, plot_metrics
from src.constants import RANDOM_STATE
from src.reference_classifier import ReferenceSGDClassifier


# def calc_metrics(
#         model: LinearClassificator | RidgeClassifier,
#         sample: DataSample,
#         verbose: bool = False,
# ) -> Metrics:
    # if isinstance(model, LinearClassificator):
    #     Y_pred = model.predict(sample)
    # else:
    #     Y_pred = model.predict(sample.X)
    
    # tp, tn, fp, fn = confusion_matrix(sample.Y.to_numpy(), Y_pred)

    # metrics = {
    #     'accuracy': (tp + tn) / (tp + tn + fp + fn),
    #     'precision': tp / (tp + fp) if tp + fp != 0 else 0
    # }

    # if verbose:
    #     print('Accuracy:', accuracy)
    #     print('Precision:', precision)

    # return accuracy, precision


def solution_corr(ds: DataSet, verbose: bool = False) -> LinearClassificator:
    if verbose:
        print('### CORR WEIGHT INIT ###\n')

    model = LinearClassificator(ds.feature_count, init_weights_method='corr', corr_sample=ds.train)
    model.fit(ds.train)
    model.evaluate(ds.test)

    if verbose:
        plot_margins(model, ds.test, 'corr')
        print('\n###\n')

    return model

    # for predict_sample in ds.train, ds.test:
    #     Y_pred = model.predict(predict_sample)
    #     tp, tn, fp, fn = confusion_matrix(predict_sample.Y.to_numpy(), Y_pred)
    #     print(tp, tn, fp, fn)

    #     print(f'{predict_sample.get_name()} Accuracy:', (tp + tn) / (tp + tn + fp + fn))
    #     try:
    #         print(f'{predict_sample.get_name()} Precision:', tp / (tp + fp))
    #     except ZeroDivisionError:
    #         print('Precision:', 0)


def solution_multistart(starts: int = 10):
    print('### MULTISTART WITH RANDOM WEIGHT INIT ###')

    max_accuracy, best_model, best_model_ds = 0, None, None # type: ignore
    for _ in range(starts):
        ds = DataSet()
        model = LinearClassificator(ds.feature_count, init_weights_method='random')
        model.sgd(ds.train)
        accuracy, _ = calc_metrics(model, ds.train)
        if accuracy > max_accuracy:
            max_accuracy, best_model, best_model_ds = accuracy, model, ds

    print_metrics(best_model, best_model_ds.test) # type: ignore
    plot_margins(best_model, best_model_ds.test, 'multistart') # type: ignore

    print('\n###', end='\n\n')


def solution_shuffling(ds: DataSet):
    print('\n### SHUFFLING OBJECT SELECTION ###\n')
    
    model = LinearClassificator(ds.feature_count, object_selection_method='shuffling')
    model.sgd(ds.train)

    print_metrics(model, ds.test)
    plot_margins(model, ds.test, 'shuffling')

    print('\n###\n')


def solution_unsure_margin(ds: DataSet):
    print('\n### UNSURE MARGIN OBJECT SELECTION ###\n')

    model = LinearClassificator(ds.feature_count, object_selection_method='unsure_first')
    model.sgd(ds.train)

    metrics = print_metrics(model, ds.test)
    plot_margins(model, ds.test, 'unsure_margin.png')

    print('\n###\n')


def reference_solution(ds: DataSet, verbose: bool = False) -> ReferenceSGDClassifier:
    if verbose:
        print('\n### REFERENCE IMPLEMENTATION ###\n')

    model = ReferenceSGDClassifier()
    model.fit(ds.train)
    model.evaluate(ds.test)

    if verbose:
        plot_margins(model, ds.test, 'reference')

    return model


def print_models_metrics(*models: Model) -> None:
    df = pd.DataFrame([{model.__class__.__name__: model.metrics.test.to_dict()} for model in models])
    # print(df)
    print([{model.__class__.__name__: model.metrics.test.to_dict()} for model in models])


def main():
    N = 100

    best_corr_model, best_ref_model = None, None
    metrics_data: dict[str, dict[str, list[float]]] = {
        'Accuracy': {},
        'Precision': {}
    }

    for i in range(N):
        ds = DataSet()

        # solution_multistart(starts=10)
        # solution_shuffling(ds)
        corr_model = solution_corr(ds)
        ref_model = reference_solution(ds)

        model: Model
        for model in corr_model, ref_model:
            model_name = model.__class__.__name__
            for metric_name, metric_val in model.metrics.test.to_dict().items(): # type: ignore
                if model_name not in metrics_data[metric_name]:
                    metrics_data[metric_name][model_name] = [metric_val]
                else:
                    metrics_data[metric_name][model_name].append(metric_val)

        if (
            best_corr_model is None or
            corr_model.metrics.test.accuracy > best_corr_model.metrics.test.accuracy # type: ignore
        ):
            best_corr_model = corr_model
        if (
            best_ref_model is None or
            ref_model.metrics.test.accuracy > best_ref_model.metrics.test.accuracy # type: ignore
        ):
            best_ref_model = ref_model

        print(f'{i}/{N}', end='\r', flush=True)

    print_models_metrics(best_corr_model, best_ref_model) # type: ignore
    plot_metrics(metrics_data)

    # print('Corr model metrics:', best_corr_model.metrics, sep='\n', end='\n\n') # type: ignore
    # print('Ref model metrics:', best_ref_model.metrics, sep='\n') # type: ignore


if __name__ == '__main__':
    np.random.seed(RANDOM_STATE)
    main()
