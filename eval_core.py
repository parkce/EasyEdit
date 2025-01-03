from easyeditor import MultimodalCaption, CoRECaptionDataset, CoREMultimodalHparams
import glob
from pathlib import Path
from statistics import mean


def print_caption(metrics):
    pre_caption_acc = 0
    post_caption_acc = mean([m['post']['rewrite_acc'].item() for m in metrics])
    # pre_caption_acc = mean([m['pre']['rewrite_acc'].item() for m in metrics])
    print(f'pre_caption_acc: {pre_caption_acc}, ft_caption_acc: {post_caption_acc}')

hparams = CoREMultimodalHparams.from_hparams('./hparams/CoRE/blip2.yaml')
editor = MultimodalCaption.from_hparams(hparams)

directory_path = Path('/kilab/data/etri/core')
# ann_paths, config
eval_ds = CoRECaptionDataset(glob.glob(str(directory_path / 'test/*json')), config=hparams)

metrics, edited_model, _ = editor.edit_dataset(
    ds=eval_ds,
    train_ds=eval_ds,
    keep_original_weight=True        
)

print_caption(metrics)
