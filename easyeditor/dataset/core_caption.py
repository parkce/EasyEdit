import os
from collections import OrderedDict
from PIL import Image, ImageFilter, ImageDraw

from easyeditor.dataset.processor.base_dataset import BaseDataset
from easyeditor.dataset.processor.blip_processors import BlipImageEvalProcessor
from easyeditor.trainer.utils import dict_to
from PIL import Image
import typing
import torch
import transformers

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
        for i, record in enumerate(self.annotation):
            
            # if record['alt'] == "":
            #     continue
            
            image_path = os.path.join(self.vis_root, record["image"])
            image = Image.open(image_path).convert("RGB")
            # blured_image = self.blur_except_box(image)

            image = self.vis_processor(image)
                      
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                # 'rephrase_prompt': record['rephrase'],
                'image': image,
                'cond': "{} >> {} || {}".format(
                    record['pred'],
                    record['alt'],
                    record['src']
                )
            }
            
            # item['locality_prompt'] = record['loc']
            # item['locality_ground_truth'] = record['loc_ans']
            
            # item['multimodal_locality_prompt'] = record['m_loc_q']
            # item['multimodal_locality_ground_truth'] = record['m_loc_a']
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
        # cond = [b['cond'] for b in batch]
        # rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch]
        # image_rephrase = [b['image_rephrase'] for b in batch]
        # loc_q = [b["locality_prompt"] for b in batch]
        # loc_a = [" " + b["locality_ground_truth"] for b in batch]
        # m_loc_image = [b['multimodal_locality_image'] for b in batch]
        # m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        # m_loc_a = [" " + b['multimodal_locality_ground_truth'] for b in batch]
        
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
        
        # edit_outer
        # edit_outer = {}
        # edit_outer['image'] = torch.stack(image, dim=0)
        # # edit_outer['text_input'] = [r + t for r, t in zip(rephrase, trg)]
        # edit_outer['labels'] = trg
        # if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
        #     edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        #     edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        # else:
        #     edit_outer['prompts_len'] = [len(self.tok.encode(r)) for r in rephrase]
        #     edit_outer['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
            
        # # edit_outer_image
        # edit_outer_image = {}
        # edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        # edit_outer_image['text_input'] = [s + t for s, t in zip(src, trg)]
        # edit_outer_image['labels'] = trg
        # if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
        #     edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        #     edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        # else:
        #     edit_outer_image['prompts_len'] = [len(self.tok.encode(s)) for s in src]
        #     edit_outer_image['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        
        # # loc
        # loc = {}
        # loc['image'] = None
        # loc['text_input'] = [q + a for q, a in zip(loc_q, loc_a)]
        # loc['labels'] = loc_a
        # if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
        #     loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        #     loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        # else:
        #     loc['prompts_len'] = [len(self.tok.encode(q)) for q in loc_q]
        #     loc['labels'] = self.tok(loc_a, return_tensors="pt",)["input_ids"]
        
        # # m_loc
        # loc_image = {}
        # loc_image['image'] = torch.stack(m_loc_image, dim=0)
        # loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
        # loc_image['labels'] = m_loc_a
        # if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
        #     loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
        #     loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        # else:
        #     loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in m_loc_q]
        #     loc_image['labels'] = self.tok(m_loc_a, return_tensors="pt",)["input_ids"]

        # cond
        # cond = self.tok(
        #     cond,
        #     return_tensors="pt",
        #     padding=True,
        #     max_length=self.max_length,
        #     truncation=True,
        # ).to(self.config.device)
        
        batch = {
            "edit_inner": edit_inner,
            # "edit_outer": edit_outer,
            # "edit_outer_image": edit_outer_image,
            # "loc": loc,
            # "loc_image": loc_image,
            # "cond": cond
        }
        return dict_to(batch, self.config.device)

    def blur_except_box(self, image, bounding_boxes, selected_box_idx):
        """
        Apply blur to all areas except the selected bounding box.

        :param image_path: Path to the input image.
        :param bounding_boxes: List of bounding boxes in the format [(x1, y1, x2, y2), ...].
        :param selected_box_idx: Index of the selected bounding box to remain unblurred.
        :param output_path: Path to save the output image.
        """
        # Open the image
        # image = Image.open(image_path)

        # Apply blur filter to the entire image
        blurred_image = image.filter(ImageFilter.GaussianBlur(15))

        # Create a mask for the unblurred area
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        # Draw the selected bounding box on the mask
        selected_box = bounding_boxes[selected_box_idx]
        draw.rectangle(selected_box, fill=255)

        # Composite the images using the mask
        final_image = Image.composite(image, blurred_image, mask)

        return final_image