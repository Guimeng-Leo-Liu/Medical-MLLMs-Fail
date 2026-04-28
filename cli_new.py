import sys
import os
import io
file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
print(dir_path)
sys.path.insert(0, dir_path)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model import *

from transformers import AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread
import torch

from PIL import Image

from typing import List, Optional, Tuple, Union
# from attention_scaling import qwen_modify
import numpy as np

class HuatuoChatbot():
    def __init__(self, model_dir, device='cuda:0'):
        self.model_dir = model_dir

        self.gen_kwargs = {
            'do_sample': False,
            'max_new_tokens': 1,
            'min_new_tokens': 1,
            # 'temperature': .2,
            # 'repetition_penalty': 1.2
            # 'output_attentions': True,
            # 'output_scores': True,
            # 'output_hidden_states': True,
            'return_dict_in_generate': True,
        }
        self.device = device
        self.init_components()
        self.history = []
        self.images = []
        self.debug = True
        self.max_image_num = 6
        

    def init_components(self):
        
        d = self.model_dir
        if 'huatuogpt-vision-7b' in d.lower():
            print(f'loading from {self.model_dir}')
            from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM
            model, loading_info = LlavaQwen2ForCausalLM.from_pretrained(self.model_dir, init_vision_encoder_from_ckpt=True, output_loading_info=True, torch_dtype=torch.bfloat16)
            missing_keys = loading_info['missing_keys'] # keys exists in model architecture but does not exist in ckpt
            unexpected_keys = loading_info['unexpected_keys'] # keys exists in ckpt but are not loaded by the model 
            assert all(['vision_tower' in k for k in unexpected_keys])

            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            self.gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
            self.gen_kwargs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
                vision_tower.vision_tower = vision_tower.vision_tower.from_pretrained(self.model_dir)
            vision_tower.to(dtype=torch.bfloat16, device=model.device)
            image_processor = vision_tower.image_processor
            
        elif 'huatuogpt' in d.lower():
            print(f'loading from {self.model_dir}')
            from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
            model, loading_info = LlavaLlamaForCausalLM.from_pretrained(self.model_dir, init_vision_encoder_from_ckpt=True, output_loading_info=True, torch_dtype=torch.bfloat16)
            missing_keys = loading_info['missing_keys'] # keys exists in model architecture but does not exist in ckpt
            unexpected_keys = loading_info['unexpected_keys'] # keys exists in ckpt but are not loaded by the model 
            assert all(['vision_tower' in k for k in unexpected_keys])

            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            self.gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
            self.gen_kwargs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
                vision_tower.vision_tower = vision_tower.vision_tower.from_pretrained(self.model_dir)
            vision_tower.to(dtype=torch.bfloat16, device=model.device)
            image_processor = vision_tower.image_processor

        else:
            raise NotImplementedError

        model.eval()
        self.model = model.to(self.device)
        self.model.config.tokenizer_padding_side = 'left'
        self.tokenizer = tokenizer
        self.processor = image_processor


    def clear_history(self,):
        self.images = []
        self.history = []

    def tokenizer_image_token(self, prompt, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None): # copied from llava
        prompt_chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def preprocess(self, data: list, return_tensors='pt'):
        '''
        [
            {
                'from': 'human',
                'value': xxx,
            },
            {
                'from': 'gpt',
                'value': xxx
            }
        ]
        '''
        if not isinstance(data, list):
            raise ValueError('must be a list')        
        return self.preprocess_huatuo(data, return_tensors=return_tensors)
    
    def preprocess_huatuo(self, convs: list, return_tensors) -> list: # tokenize and concat the coversations
        input_ids = None
        convs = [ conv for conv in convs if conv['value'] is not None]
        round_num = len(convs)//2

        for ind in range(round_num):
            h = convs[ind*2]['value'].strip()
            h = f"<|user|>\n{h}\n" 

            g = convs[ind*2+1]['value']
            g = f"<|assistant|>\n{g} \n"

            cur_input_ids = self.tokenizer_image_token(prompt=h, return_tensors=return_tensors)

            if input_ids is None:
                input_ids = cur_input_ids
            else:
                input_ids = torch.cat([input_ids, cur_input_ids])
            
            cur_input_ids = self.tokenizer(g, add_special_tokens= False, truncation=True, return_tensors='pt').input_ids[0]
            input_ids = torch.cat([input_ids, cur_input_ids])
        
        h = convs[-1]['value'].strip()
        h = f"<|user|>\n{h}\n<|assistant|>\n"
        cur_input_ids = self.tokenizer_image_token(prompt=h, return_tensors=return_tensors)

        if input_ids is None:
            input_ids = cur_input_ids
        else:
            input_ids = torch.cat([input_ids, cur_input_ids])
        
        if self.debug:
            self.debug = False

        return input_ids


    def input_moderation(self, t: str):
        blacklist = ['<image>', '<s>', '</s>']
        for b in blacklist:
            t = t.replace(b, '')
        return t
    
    def insert_image_placeholder(self, t, num_images, placeholder='<image>', sep='\n'):
        for _ in range(num_images):
            t = f"{placeholder}{sep}" + t

        return t
    
    def get_conv(self, text):
        ret = []
        if self.history is None:
            self.history = []
        
        for conv in self.history:
            ret.append({'from': 'human', 'value': conv[0]})
            ret.append({'from': 'gpt', 'value': conv[1]})

        ret.append({'from': 'human', 'value': text})
        ret.append({'from': 'gpt', 'value': None})

        return ret

    def get_conv_without_history(self, text):
        ret = []

        ret.append({'from': 'human', 'value': text})
        ret.append({'from': 'gpt', 'value': None})

        return ret
    
    def get_image_tensors(self, images):
        list_image_tensors = []
        crop_size = self.processor.crop_size
        processor = self.processor
        for fp in images:
            if fp is None: # None is used as a placeholder
                continue
            elif isinstance(fp, str):
                image = Image.open(fp).convert('RGB')
            elif isinstance(fp, Image.Image):
                image = fp # already an image
            elif isinstance(fp, bytes):
                image = Image.open(io.BytesIO(fp)).convert('RGB')
            else:
                raise TypeError(f'Unsupported type {type(fp)}')

            if True or self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0] # a tensor
            list_image_tensors.append(image.to(self.device))
        if len(list_image_tensors) == 0:
            list_image_tensors.append(torch.zeros(3, crop_size['height'], crop_size['width']).to(self.device))
        return list_image_tensors

    def inference(self, text, images=None, return_att_map=False, return_hidden_states=False):
        '''
        text: str
        images: list[str] or list[bytes] or list[PIL.Image.Image]
        '''
        
        # image
        if images is None:
            images = []

        if isinstance(images,str):
            images = [images]

        valid_images = []
        for img in images:
            try:
                if isinstance(img, str):
                    Image.open(img).convert('RGB') # make sure that the path exists
                elif isinstance(img, bytes):
                    Image.open(io.BytesIO(img)).convert('RGB') # validate bytes can be opened as image
                valid_images.append(img)
            except:
                print(f'{img} This image is wrong.')
                continue
        images = valid_images

        if len(valid_images) > self.max_image_num:
            images = images[:self.max_image_num]

        # text
        text = self.input_moderation(text)
        text = self.insert_image_placeholder(text, len(images) if None not in images else 0)

        conv = self.get_conv_without_history(text)
        input_ids = self.preprocess(conv, return_tensors='pt').unsqueeze(0).to(self.device)

        if len(images) > 0:
            list_image_tensors = self.get_image_tensors(images)
            image_tensors = torch.stack(list_image_tensors).to(dtype=torch.bfloat16).to(self.device)
        else:
            image_tensors = None

        with torch.inference_mode():
            output_ids = None
            if return_att_map:
                output_ids = self.model(
                    input_ids,
                    images=image_tensors,
                    use_cache=True,
                    attention_mask=None,
                    output_attentions=True
                )
                return output_ids, input_ids, None
            outputs = self.model.generate(
                input_ids,
                images=image_tensors,
                use_cache=True,
                output_hidden_states=return_hidden_states,
                **self.gen_kwargs)

        if not return_hidden_states:
            return output_ids, input_ids, self.tokenizer.decode(outputs['sequences'][0],skip_special_tokens=True).strip()
        else:
            return output_ids, input_ids, self.tokenizer.decode(outputs['sequences'][0],skip_special_tokens=True).strip(), outputs
    

    def chat(self, text: str, images: list[str]=None, ):
        '''
        images: list[str], images for this round
        text: str
        '''
        text = self.input_moderation(text)
        if text == '':
            return 'Please type in something'

        if isinstance(images, str) or isinstance(images, Image.Image):
            images = [images]
        
        valid_images = []
        if images is None:
            images = []
        
        for img in images:
            try:
                if isinstance(img, str):
                    Image.open(img).convert('RGB') # make sure that the path exists
                valid_images.append(img)
            except:
                continue

        images = valid_images

        self.images.extend(images)


        assert len(images) < self.max_image_num, f'at most {self.max_image_num} images'

        text = self.insert_image_placeholder(text, len(images) if None not in images else 0)
        # make conv
        conv = self.get_conv(text)
        # make input ids
        input_ids = self.preprocess(conv, return_tensors='pt').unsqueeze(0).to(self.device)

        if len(self.images) > 0:
            list_image_tensors = self.get_image_tensors(self.images)
            image_tensors = torch.stack(list_image_tensors)
        else:
            image_tensors = None

        streamer = TextIteratorStreamer(self.tokenizer,skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(inputs=input_ids,images=image_tensors.to(dtype=torch.bfloat16) if image_tensors is not None else image_tensors, streamer=streamer,use_cache=True,**self.gen_kwargs)


        with torch.inference_mode():
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            generated_text = ''
            sep = self.tokenizer.convert_ids_to_tokens(self.tokenizer.eos_token_id)
            for new_text in streamer:
                if sep in new_text:
                    new_text = self.remove_overlap(generated_text,new_text[:-len(sep)])
                    for char in new_text:
                        generated_text += char
                        print(char,end='',flush = True)
                    break
                for char in new_text:
                    generated_text += char
                    print(char,end='',flush = True)
        answer = generated_text

        self.history.append([text, answer])

        return answer
    
    def prepare_input(
            self, 
            text, 
            images=None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
        ):
        '''
        text: str
        images: list[str] or list[bytes] or list[PIL.Image.Image]
        '''
        
        # image
        if images is None:
            images = []

        if isinstance(images,str):
            images = [images]

        valid_images = []
        for img in images:
            try:
                if isinstance(img, str):
                    Image.open(img).convert('RGB') # make sure that the path exists
                elif isinstance(img, bytes):
                    Image.open(io.BytesIO(img)).convert('RGB') # validate bytes can be opened as image
                valid_images.append(img)
            except:
                print(f'{img} This image is wrong.')
                continue
        images = valid_images

        if len(valid_images) > self.max_image_num:
            images = images[:self.max_image_num]

        # text
        text = self.input_moderation(text)
        text = self.insert_image_placeholder(text, len(images) if None not in images else 0)

        conv = self.get_conv_without_history(text)
        input_ids = self.preprocess(conv, return_tensors='pt').unsqueeze(0).to(self.device)

        if len(images) > 0:
            list_image_tensors = self.get_image_tensors(images)
            image_tensors = torch.stack(list_image_tensors).to(dtype=torch.bfloat16).to(self.device)
        else:
            image_tensors = None

        with torch.inference_mode():
            (
                _,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.model.prepare_inputs_labels_for_multimodal_new(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                image_tensors
            )
        return input_ids, inputs_embeds
    
    def remove_wrapper_llava(self, model, hooks):
        for i, hook in hooks:
            model.model.layers[i].self_attn.forward =hook
        

    def inference_with_attention_amplify(self, text, images, method_name):
        '''
        text: str
        images: list[str]
        '''
        
        # image
        if images is None:
            images = []

        if isinstance(images,str):
            images = [images]

        valid_images = []
        for img in images:
            try:
                if isinstance(img, str):
                    Image.open(img).convert('RGB') # make sure that the path exists
                elif isinstance(img, bytes):
                    Image.open(io.BytesIO(img)).convert('RGB')
                valid_images.append(img)
            except:
                print(f'{img} This image is wrong.')
                continue
        images = valid_images
        if len(valid_images) > self.max_image_num:
            images = images[:self.max_image_num]

        # text
        text = self.input_moderation(text)
        text = self.insert_image_placeholder(text, len(images) if None not in images else 0)

        conv = self.get_conv_without_history(text)
        input_ids = self.preprocess(conv, return_tensors='pt').unsqueeze(0).to(self.device)

        if len(images) > 0:
            list_image_tensors = self.get_image_tensors(images)
            image_tensors = torch.stack(list_image_tensors).to(dtype=torch.bfloat16).to(self.device)
        else:
            image_tensors = None

        if method_name == 'ScaleVis':
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensors,
                    use_cache=True,
                    output_scores=True,
                    **self.gen_kwargs)
                confidence = np.round(float(max(torch.nn.functional.softmax(output_ids['scores'][0], dim=-1)[0])), 2)
        else:
            confidence = 0.0

        image_range = list(range(torch.where(input_ids[0]==-200)[0], torch.where(input_ids[0]==-200)[0] + 576))

        with torch.inference_mode():

            block_attn_hooks = qwen_modify(self.model, method_name, image_range, confidence)
            output_ids = self.model.generate(
                input_ids, 
                images=image_tensors,
                use_cache=True,
                output_attentions=True, # make sure this is true so that the model use eager attention layer
                **self.gen_kwargs)
            self.remove_wrapper_llava(self.model, block_attn_hooks)

        return self.tokenizer.decode(output_ids['sequences'][0],skip_special_tokens=True).strip()

        # answers = []
        # for output_id in output_ids:
        #     answers.append(self.tokenizer.decode(output_id, skip_special_tokens=True).strip())
        # return answers


if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Args of Data Preprocess')

    parser.add_argument('--model_dir', default='', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    args = parser.parse_args()

    bot = HuatuoChatbot(args.model_dir, args.device)

    # test
    # print(bot.inference('what show in this picture?',['./output.png']))
    # print(bot.inference('hi'))

    while True:
        images = input('images, split by ",": ')
        images = [i.strip() for i in images.split(',') if len(i.strip()) > 1 ]
        text = input('USER ("clear" to clear history, "q" to exit): ')
        if text.lower() in ['q', 'quit']:
            exit()

        if text.lower() == 'clear':
            bot.history = []
            bot.images = []
            continue

        answer = bot.chat(images=images, text=text)

        images = None # already in the history

        print()
        print(f'GPT: {answer}')
        print()
