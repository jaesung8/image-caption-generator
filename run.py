import datetime

import click
import pytz

from src.predictor import Predictor
from src.trainer import Trainer


@click.group()
def cli():
    pass


@click.command()
def train():
    hyperparameters = {
        "learning_rate": 1e-4,
        "gamma": 0.98,
        "num_epochs": 100,
        "epoch_interval": 10,
    }
    trainer = Trainer(**hyperparameters)
    trainer.train()


@click.command()
def predict():
    pass


cli.add_command(train)
cli.add_command(predict)
cli()