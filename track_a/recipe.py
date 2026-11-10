"""Track A recipe: the starter recipe, with its city bias corrected at the end.

Everything is the starter recipe (same augmentation, sampler, optimiser,
schedule, loss, batch size). The one change is in final_model: five numbers
are subtracted from the last layer's biases, so that the model stops putting
36% of the photos in Tokyo when each city is 20% of them.

Why five numbers and not a threshold at the end: the grader exports the model
to ONNX and scores the file, so the correction has to be inside the model. The
last layer's bias is part of it, and shifting it leaves the network itself
untouched -- the same ResNet18 comes back, as official.py requires.

Where the numbers come from: the corrected score for city i is
    logit_i - tau * log(n_i / N)
with n_i the number of training photos of city i and N their total. Training
draws Tokyo photos 2.6 times as often as Busan ones, so the model's scores
inherit that imbalance, while the score is the mean over the five cities.
tau = 1 is the usual strength; on the validation photos of the starter recipe
it moves 0.4987 to 0.5098 (see scripts/prior_fix.py). The counts below are the
starter training folders; this recipe is released without selection.csv, so
they do not move.
"""

import math

import torch
import torch.nn.functional as F
from torchvision import transforms as T

BATCH_SIZE = 64

# Training photos per city, in the order of rules.CITIES, when every training
# photo is used. This is what the recipe below corrects for.
CITY_COUNTS = [11274, 5191, 13347, 7200, 7245]
CORRECTION_STRENGTH = 1.0


def train_transform():
    """A function that turns one training photo (a PIL image) into a 224 x 224 one."""
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.35, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
    ])


def sample_weights(cities, paths):
    """One pass through the selection per round, in a new random order."""
    return None


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)


def make_scheduler(optimizer, total_steps):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, total_steps=total_steps, pct_start=0.15)


def mix(images, labels):
    """No mixing: the photos go to the model as they are."""
    return images, F.one_hot(labels.long(), 5).float()


def loss_fn(logits, targets):
    return F.cross_entropy(logits, targets, label_smoothing=0.1)


def on_step(step, total_steps, model, optimizer):
    """Called after every step, with the step number from 0. Returns nothing."""


def final_model(model):
    """The model to hand in. Still the same ResNet18, with corrected biases."""
    total = sum(CITY_COUNTS)
    log_prior = [math.log(n / total) for n in CITY_COUNTS]
    middle = sum(log_prior) / len(log_prior)
    shift = [(l - middle) * CORRECTION_STRENGTH for l in log_prior]
    with torch.no_grad():
        bias = model.fc.bias
        bias -= torch.tensor(shift, dtype=bias.dtype, device=bias.device)
    return model