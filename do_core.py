from easyeditor import MultimodalTrainer, CoRECaptionDataset, CoREMultimodalTrainingHparams


training_hparams = CoREMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/CoRE/blip2.yaml')

train_ds = CoRECaptionDataset('/kilab/data/editing-data/caption/core_train.json', config=training_hparams)
eval_ds = CoRECaptionDataset('/kilab/data/editing-data/caption/core_eval.json', config=training_hparams)

trainer = MultimodalTrainer(config=training_hparams, train_set=train_ds,val_set=eval_ds)
trainer.run()