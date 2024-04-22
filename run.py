import datetime
import os
import random

import click
import pytz

from src.predictor import Predictor, AttentionPredictor
from src.trainer import Trainer, AttentionTrainer
from src.utils import visualize_att


@click.group()
def cli():
    pass


@click.command()
@click.argument("model_type", type=click.STRING)
@click.argument("device_number", type=click.STRING)
def train(model_type, device_number):
    hyperparameters = {
        "learning_rate": 1e-4,
        "gamma": 0.96,
        "num_epochs": 120,
        "epoch_interval": 10,
        "alpha_lambda": 1.0,
        "device_number": device_number,
    }
    if model_type == "vanila":
        trainer = Trainer(**hyperparameters)
    elif model_type == "attention":
        trainer = AttentionTrainer(**hyperparameters)
    else:
        raise ValueError("Unknown model")

    trainer.train()


@click.command()
@click.argument("model_type", type=click.STRING)
@click.argument("device_number", type=click.STRING)
def predict(model_type, device_number):
    hyperparameters = {
        "device_number": device_number,
    }

    predictor = AttentionPredictor(**hyperparameters)
    words_list, alphas_list, path_cpation_list, top_idices = predictor.predict()
    for i in top_idices:
        # random_index = random.randint(0, len(words_list))
        visualize_att(path_cpation_list[i], alphas_list[i], words_list[i])


cli.add_command(train)
cli.add_command(predict)
cli()