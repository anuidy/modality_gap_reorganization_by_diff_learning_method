"""Raw encodings with GPU JPEG/preprocessing and compact CPU feature caches."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.io import decode_jpeg,ImageReadMode
from torchvision.transforms import functional as TF

from embeddings.artifact import sha256_file


class RawEncoder:
    def __init__(self,adapter,model,batch_size=32,workers=2,image_cache=None,cache_source_root=None):
        if model not in ('clip','beit3','vista'):raise ValueError('Current A/B diagnostic supports the three main models')
        self.adapter,self.model,self.batch_size=adapter,model,batch_size
        self.device=adapter.device;self.preprocess=adapter.model.preprocess_val if model=='vista' else adapter.preprocess
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.images={};self.texts={};self.joints={};self.image_hashes={}
        self.stats={'gpu_jpeg_decoded':0,'cpu_decode_fallback':0,'verified_cached_images':0}
        self.cache=None;self.cache_identity=None;self.cache_source_root=cache_source_root
        if image_cache is not None and (image_cache/'complete.json').is_file():
            complete=json.loads((image_cache/'complete.json').read_text())
            if sha256_file(image_cache/'index.json')!=complete['files']['index.json']:raise ValueError('Image cache index changed')
            index=json.loads((image_cache/'index.json').read_text())
            if index['preprocess']!='gpu_nvjpeg_tensor_bicubic_v1':raise ValueError('Unsupported pixel cache protocol')
            resize=next(x for x in self.preprocess.transforms if isinstance(x,transforms.Resize))
            crop=next((x for x in self.preprocess.transforms if isinstance(x,transforms.CenterCrop)),None)
            family='crop224' if resize.size==224 and crop is not None else 'warp224'
            if family=='warp224' and tuple(resize.size)!=(224,224):raise ValueError('Unexpected resize geometry')
            cache_path=image_cache/(family+'.npy')
            if sha256_file(cache_path)!=complete['files'][family+'.npy']:raise ValueError('Image pixel cache changed')
            self.cache=np.load(cache_path,mmap_mode='r');self.cache_lookup={name:i for i,name in enumerate(index['names'])}
            self.cache_sources={row['path']:row['sha256'] for row in index['images']}
            self.norm=next(x for x in self.preprocess.transforms if isinstance(x,transforms.Normalize))
            self.cache_identity={'complete_sha256':sha256_file(image_cache/'complete.json'),'family':family}

    def close(self):self.pool.shutdown()

    def verify_image(self,path):
        key=str(Path(path).resolve())
        if key not in self.image_hashes:self.image_hashes[key]=sha256_file(Path(key))
        return key

    def transform(self,image):
        for operation in self.preprocess.transforms:
            if isinstance(operation,transforms.Resize):image=TF.resize(image,operation.size,operation.interpolation,antialias=True)
            elif isinstance(operation,transforms.CenterCrop):image=TF.center_crop(image,operation.size)
            elif isinstance(operation,transforms.ToTensor):image=TF.convert_image_dtype(image,torch.float32)
            elif isinstance(operation,transforms.Normalize):image=TF.normalize(image,operation.mean,operation.std)
            elif getattr(operation,'__name__','') in ('_convert_image_to_rgb','_convert_to_rgb'):pass
            else:raise ValueError('Unsupported tensor preprocessing operation: '+repr(operation))
        return image

    def image_batch(self,paths):
        paths=[self.verify_image(p) for p in paths]
        keys=[]
        if self.cache is not None:
            for path in paths:
                try:key=Path(path).relative_to(self.cache_source_root.resolve()).as_posix()
                except ValueError:break
                if key not in self.cache_lookup:break
                if self.image_hashes[path]!=self.cache_sources[key]:raise ValueError('Source image differs from verified pixel cache')
                keys.append(key)
        if len(keys)==len(paths) and self.cache is not None:
            value=np.array(self.cache[[self.cache_lookup[k] for k in keys]],copy=True)
            value=torch.from_numpy(value).to(self.device)
            self.stats['verified_cached_images']+=len(paths)
            return TF.normalize(TF.convert_image_dtype(value,torch.float32),self.norm.mean,self.norm.std)
        raw=list(self.pool.map(lambda p:Path(p).read_bytes(),paths));decoded={}
        if self.device.type=='cuda':
            jpeg=[i for i,data in enumerate(raw) if data[:2]==b'\xff\xd8']
            if jpeg:
                try:
                    values=decode_jpeg([torch.from_numpy(np.frombuffer(raw[i],np.uint8).copy()) for i in jpeg],
                                       mode=ImageReadMode.RGB,device=self.device)
                    decoded.update(zip(jpeg,values,strict=True));self.stats['gpu_jpeg_decoded']+=len(jpeg)
                except RuntimeError as error:
                    if 'out of memory' in str(error).lower():raise
                    # Decode unsupported/corrupt cases individually; never replace an image by black pixels.
        output=[]
        for i,path in enumerate(paths):
            if i not in decoded:
                with Image.open(path) as image:value=TF.pil_to_tensor(image.convert('RGB')).to(self.device)
                self.stats['cpu_decode_fallback']+=1
            else:value=decoded[i]
            output.append(self.transform(value))
        return torch.stack(output)

    @staticmethod
    def cpu_rows(value):
        rows=value.detach().float().cpu().contiguous().numpy().copy()
        if not np.isfinite(rows).all():raise ValueError('Nonfinite raw embedding')
        return rows

    @torch.inference_mode()
    def image_rows(self,paths):
        paths=[str(Path(p).resolve()) for p in paths]
        missing=[p for p in dict.fromkeys(paths) if p not in self.images]
        for start in range(0,len(missing),self.batch_size):
            chunk=missing[start:start+self.batch_size];batch=self.image_batch(chunk);a=self.adapter
            with a._autocast():
                if self.model in ('clip','vista'):vectors=a.model.encode_image(batch)
                else:vectors=a.model.vision_head(a.model.beit3(textual_tokens=None,visual_tokens=batch,text_padding_position=None)['encoder_out'][:,0,:])
            rows=self.cpu_rows(vectors);self.images.update(zip(chunk,rows,strict=True))
        return np.stack([self.images[p] for p in paths])

    @torch.inference_mode()
    def text_rows(self,texts):
        missing=[t for t in dict.fromkeys(texts) if t not in self.texts]
        for start in range(0,len(missing),self.batch_size):
            chunk=missing[start:start+self.batch_size];rows=self.cpu_rows(self.adapter.encode_text(chunk))
            self.texts.update(zip(chunk,rows,strict=True))
        return np.stack([self.texts[t] for t in texts])

    @torch.inference_mode()
    def joint_rows(self,paths,texts):
        if self.model!='vista':return self.image_rows(paths)+self.text_rows(texts)
        keys=[(str(Path(p).resolve()),t) for p,t in zip(paths,texts,strict=True)]
        missing=[k for k in dict.fromkeys(keys) if k not in self.joints];a=self.adapter
        for start in range(0,len(missing),self.batch_size):
            chunk=missing[start:start+self.batch_size];images=self.image_batch([p for p,t in chunk])
            tokens=a.model.tokenizer([t for p,t in chunk],padding=True,truncation=True,
                max_length=max(1,512-a._image_token_count()),return_tensors='pt').to(self.device)
            with a._autocast():vectors=a.model.encode_mm(images,tokens)
            self.joints.update(zip(chunk,self.cpu_rows(vectors),strict=True))
        return np.stack([self.joints[k] for k in keys])

    def paired(self,paths,texts):
        image=self.image_rows(paths);text=self.text_rows(texts)
        return {'I':image,'T':text,'IT':self.joint_rows(paths,texts) if self.model=='vista' else image+text}

    def fused(self,pairs,image_root):
        image_keys=[str((image_root/i).resolve()) for t,i in pairs if i and not(self.model=='vista' and t)]
        text_keys=[t for t,i in pairs if t and not(self.model=='vista' and i)]
        if image_keys:self.image_rows(image_keys)
        if text_keys:self.text_rows(text_keys)
        joint_keys=[(str((image_root/i).resolve()),t) for t,i in pairs if self.model=='vista' and t and i]
        if joint_keys:self.joint_rows([p for p,t in joint_keys],[t for p,t in joint_keys])
        output=[]
        for text,image in pairs:
            path=str((image_root/image).resolve()) if image else None
            if self.model=='vista' and text and image:value=self.joints[(path,text)]
            elif text and image:value=self.images[path]+self.texts[text]
            elif text:value=self.texts[text]
            else:value=self.images[path]
            output.append(value)
        return np.stack(output).astype(np.float32)
