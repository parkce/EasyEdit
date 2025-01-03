from easyeditor import CoREMultimodalTrainer, CoRECaptionDataset, CoREMultimodalTrainingHparams
import glob
from pathlib import Path

training_hparams = CoREMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/CoRE/blip2.yaml')

directory_path = Path('/kilab/data/etri/core')
# ann_paths, config
train_ds = CoRECaptionDataset(glob.glob(str(directory_path / 'train/*json')), config=training_hparams)
eval_ds = CoRECaptionDataset(glob.glob(str(directory_path / 'test/*json')), config=training_hparams)

trainer = CoREMultimodalTrainer(config=training_hparams, train_set=train_ds, val_set=eval_ds)
trainer.run()