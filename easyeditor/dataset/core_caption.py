import os
from PIL import Image, ImageFilter, ImageDraw
from easyeditor.dataset.processor.base_dataset import BaseDataset
from easyeditor.dataset.processor.blip_processors import BlipImageEvalProcessor
from easyeditor.trainer.utils import dict_to
from PIL import Image
import typing
import torch
import transformers
from tqdm import tqdm


class CoRECaptionDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.core_image
        # rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, None, data_dir)

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer:"

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]  
        for i, record in tqdm(enumerate(self.annotation), total=len(self.annotation)):
            image_id = next(iter(record.keys()))
            image_path = os.path.join(self.vis_root, image_id+".jpg")
            image = Image.open(image_path).convert("RGB")
            image_idx = self.vis_processor(image)
            
            # for blured_image
            for region in record[image_id]['regions'][:1]:
                bbox = (region['x'], region['y'], region['x']+region['width'], region['y']+region['height'])
            
                blured_image = self.blur_except_box(image, bbox)
                blured_image_idx = self.vis_processor(blured_image)
                      
                for caption in region['captions'][:1]:
                    item = {
                        'target': caption['caption'],
                        'image': image_idx,
                        'blured_image': blured_image_idx,
                    }
                    data.append(item)
            
        # if size is not None:
        #     data = data[:size]        
        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [" " + b['target'] for b in batch]
        image = [b['image'] for b in batch]
        
        # edit_inner
        edit_inner = {}
        edit_inner['image'] = torch.stack(image, dim=0)
        edit_inner['text_input'] = [s + t for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_inner['prompts_len'] = [len(self.tok.encode(s)) for s in src]
            edit_inner['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        
        
        batch = {
            "edit_inner": edit_inner,
        }
        return dict_to(batch, self.config.device)

    def blur_except_box(self, image, bounding_box):
        """
        Apply blur to all areas except the selected bounding box.

        :param image_path: Path to the input image.
        :param bounding_boxes: List of bounding boxes in the format [(x1, y1, x2, y2), ...].
        :param output_path: Path to save the output image.
        """
        # Apply blur filter to the entire image
        blurred_image = image.filter(ImageFilter.GaussianBlur(15))

        # Create a mask for the unblurred area
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        # Draw the selected bounding box on the mask
        draw.rectangle(bounding_box, fill=255)

        # Composite the images using the mask
        final_image = Image.composite(image, blurred_image, mask)

        return final_image