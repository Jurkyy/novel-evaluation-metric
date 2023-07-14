import importlib as importer
import numpy as np
import yaml
from pathlib import Path

with open("config.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

for configuration in config["configurations"]:

    # get data
    dataset = configuration["dataset"]

    Path(f"results/{dataset}/training_history/").mkdir(parents=True, exist_ok=True)
    Path(f"results/{dataset}/trained_models/").mkdir(parents=True, exist_ok=True)
    Path(f"results/{dataset}/model_predictions/").mkdir(parents=True, exist_ok=True)
    Path(f"results/{dataset}/plot_data/").mkdir(parents=True, exist_ok=True)
    Path(f"results/{dataset}/fig/").mkdir(parents=True, exist_ok=True)

    # start the training
    for loss_func in configuration["loss_function"]:
        for nr_epochs in configuration["nr_epochs"]:
            for batch_size in configuration["batch_sizes"]:
                getattr(importer.import_module(f"datasets.{dataset}.cnn_architecture"),
                        "run")(model_name=configuration['model'], loss_func=loss_func,
                               metrics_names=configuration['metrics'], nb_epochs=nr_epochs, batch_size=batch_size,
                               learning_rate=configuration['learning_rate'],
                               reshape_x_attack=configuration['reshape_x_attack'],
                               early_stopping=configuration['early_stopping'],
                               early_stopping_patience=configuration['early_stopping_patience'])
