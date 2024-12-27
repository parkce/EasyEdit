from easyeditor import MultimodalTrainer, CaptionDataset
from .trainer.training_hparams.core_multimodal_training import CoREMultimodalTrainingHparams


training_hparams = CoREMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/SERAC/blip2-gpt2.yaml')

train_ds = CaptionDataset('/kilab/data/editing-data/caption/core_train.json', config=training_hparams)
eval_ds = CaptionDataset('/kilab/data/editing-data/caption/core_eval.json', config=training_hparams)

trainer = MultimodalTrainer(config=training_hparams, train_set=train_ds,val_set=eval_ds)
trainer.run()