from easyeditor import MultimodalTrainer, CoRECaptionDataset, CoREMultimodalTrainingHparams
import glob
from pathlib import Path

training_hparams = CoREMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/CoRE/blip2.yaml')

directory_path = Path('/path/to/your/directory')
# ann_paths, config
train_ds = CoRECaptionDataset(glob.glob(directory_path / 'train/*json'), config=training_hparams)
eval_ds = CoRECaptionDataset(glob.glob(directory_path / 'test/*json'), config=training_hparams)

trainer = MultimodalTrainer(config=training_hparams, train_set=train_ds,val_set=eval_ds)
trainer.run()