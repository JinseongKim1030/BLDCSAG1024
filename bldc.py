import torch
import cv2
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from tkinter import *
from transformers import SamModel, SamProcessor
from PIL import Image
from diffusers import DDIMScheduler
from diffusers import (
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)

class BlendedLatentDiffusionWithControlNet:
    def __init__(self, prompt, init_image, mask, model_path, controlnet_model_path, batch_size, blending_start_percentage, device, output_path):
        self.prompt = prompt
        self.init_image = init_image
        self.mask = mask
        self.model_path = model_path
        self.controlnet_model_path = controlnet_model_path
        self.batch_size = batch_size
        self.blending_start_percentage = blending_start_percentage
        self.device = device
        self.output_path = output_path
        self.load_models()

    def load_models(self):
        # ControlNet 모델 로드
        self.controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model_path, torch_dtype=torch.float16
        ).to(self.device)

        # ControlNet이 포함된 Stable Diffusion 파이프라인 로드
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_path,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
        ).to(self.device)

        self.vae = self.pipe.vae.to(self.device)
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.to(self.device)
        self.unet = self.pipe.unet.to(self.device)
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

    @torch.no_grad()
    def edit_image(
        self,
        image_path,
        mask_path,
        prompts,
        height=512,
        width=512,
        num_inference_steps=70,
        guidance_scale=7.0,
        generator=None,
        blending_percentage=0.05,
    ):
        # 랜덤 시드 설정
        if generator is None:
            generator = torch.Generator(device=self.device)  # self.device로 변경
            generator.manual_seed(random.randint(1, 2147483647))

        #1 주어진 프롬프트의 개수를 기준으로 배치 크기 설정
        batch_size = len(prompts)

        #2 이미지 로드 및 리사이즈
        # 입력 경로에서 이미지 오픈
        image = Image.open(image_path)
        # 지정된 높이(height)와 너비(width)로 리사이즈
        image = image.resize((height, width), Image.BILINEAR)
        # 이미지를 넘파이 배열로 변환 (RGB 채널만 사용)
        image = np.array(image)[:, :, :3]

        #3 Canny Edge 이미지 생성
        # 입력 이미지의 Canny Edge 이미지 생성
        canny_image = self._create_canny_image(image)
        Image.fromarray(canny_image).save('canny.png')
        # 이미지를 controlnet_cond로 변환하여 ControlNet 모델이 사용할 수 있는 형태로 준비
        controlnet_cond = self._prepare_control_image(canny_image)

        #4 원본 이미지를 latent space로 변환
        source_latents = self._image2latent(image)

        #5 마스크 로드 및 처리
        latent_mask, org_mask = self._read_mask(mask_path)

        #6 텍스트 임베딩 생성
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # 네거티브 프롬프트 임베딩 생성
        max_length = text_input.input_ids.shape[-1]
        negative_prompt = "low quality, blurry, distorted, out of focus, low resolution, bad anatomy, bad proportions, extra limbs, missing limbs"
        uncond_input = self.tokenizer(
            [negative_prompt] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

        # Initial Latent 설정
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=torch.float16
        )
        #latents = latents * latent_mask

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Ensure tensors have correct batch sizes
        source_latents = source_latents.repeat(batch_size, 1, 1, 1)
        latent_mask = latent_mask.repeat(batch_size, 1, 1, 1)
        controlnet_cond = controlnet_cond.repeat(batch_size, 1, 1, 1).to(self.device).half()

        for t in timesteps[int(len(timesteps) * blending_percentage):]:
            # Scale latent_model_input
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # Duplicate latent_model_input and controlnet_cond for CFG
            latent_model_input = torch.cat([latent_model_input] * 2, dim=0)
            controlnet_cond_in = torch.cat([controlnet_cond] * 2, dim=0)

            # ControlNet
            controlnet_output = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=controlnet_cond_in,
                return_dict=False,
            )

            # UNet
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=controlnet_output[0],
                mid_block_additional_residual=controlnet_output[1],
            ).sample

            # Conditional & Unconditional predictions separation and CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # Update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Blending with mask
            noise_source_latents = self.scheduler.add_noise(
                source_latents,
                torch.randn(latents.shape, device=latents.device, generator=generator, dtype=latents.dtype),
                t
            )
            kernel = np.ones((1, 1), np.uint8)

            eroded_masks = torch.empty_like(latent_mask)

            # 각 채널에 대해 침식 수행
            for i in range(latent_mask.size(1)):
                # [batch_size, channels, height, width]에서 채널별로 [height, width] 부분만 가져오기
                mask_np = latent_mask[0, i].cpu().numpy()  # PyTorch 텐서를 NumPy 배열로 변환

                # float16 타입을 uint8로 변환 및 값 범위 조정
                mask_np_uint8 = (mask_np * 255).astype(np.uint8)

                # OpenCV로 erosion 수행
                eroded_np_uint8 = cv2.dilate(mask_np_uint8, kernel, iterations=1)

                # 다시 원래 타입과 값 범위로 변환
                eroded_np = eroded_np_uint8.astype(np.float32) / 255.0

                # PyTorch 텐서로 변환하여 저장
                eroded_masks[0, i] = torch.from_numpy(eroded_np).to(latent_mask.dtype)

            latents = latents * eroded_masks + noise_source_latents * (1 - latent_mask)
        # Decode the latents to images
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        # Post-processing and return
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images

    @torch.no_grad()
    def _image2latent(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        image = image.half()
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215

        return latents

    def _read_mask(self, mask_path: str, dest_size=(64, 64)):
        org_mask = Image.open(mask_path).convert("L")
        mask = org_mask.resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask[np.newaxis, np.newaxis, ...]
        mask = torch.from_numpy(mask).half().to(self.device)

        return mask, org_mask

    def _create_canny_image(self, image):
        # Create Canny Edge image using OpenCV
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        edges = cv2.resize(edges, (512, 512), interpolation=cv2.INTER_LINEAR)
        return edges

    def _prepare_control_image(self, image):
        if not isinstance(image, torch.Tensor):
            image = np.array(image)
            if len(image.shape) == 2:
                image = image[:, :, None]
            image = image.transpose(2, 0, 1)  # (C, H, W)
            if image.shape[0] == 1:
                image = np.repeat(image, repeats=3, axis=0)  # Increase channels from 1 to 3
            image = image / 255.0
            image = torch.from_numpy(image).float()
        return image.to(self.device).half()
    
    
class SamImageProcessor:
    def __init__(self, img_path, model_name="facebook/sam-vit-huge"):
        self.img_path = img_path
        self.device, self.torch_dtype = self._set_device()
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_name)
        self.pf_image = self._load_image()

    def _set_device(self):
        if sys.platform == "darwin":
            device = torch.device("mps")
            torch_dtype = torch.float32
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            torch_dtype = torch.float16
        else:
            device = torch.device("cpu")
            torch_dtype = torch.float32
        return device, torch_dtype

    def _load_image(self):
        pf_image = Image.open(self.img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        pf_image.save('original.png')
        return pf_image

    def process_image(self):
        input_points = [[[self.pf_image.size[0] // 2, self.pf_image.size[1] // 2]]]  # 2D location of the center
        inputs = self.processor(self.pf_image, input_points=input_points, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return inputs, outputs

    def post_process_masks(self, inputs, outputs):
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores
        return masks, scores

    def extract_max_score_mask(self, masks, scores):
        max_score_index = torch.argmax(scores[0][0])  # 가장 큰 스코어 값의 인덱스
        max_score = torch.max(scores[0][0]).item()
        max_score_mask = masks[0][:, 2, :, :]  # 가장 스코어가 큰 마스크만 가져오기
        return max_score_mask

    def save_mask_image(self, max_score_mask):
        color = np.array([0, 0, 0, 0.9])  # R, G, B, 투명도 (0:완전 투명, 1은 불투명) 값
        h, w = max_score_mask.shape[-2:]
        mask_image = np.where(max_score_mask.reshape(h, w, 1) != 0, color.reshape(1, 1, -1), [1, 1, 1, 1])
        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8), 'RGBA')
        mask_image_pil.save('mask.png')

    def save_masked_image(self, max_score_mask):
        max_score_mask_resized = max_score_mask.squeeze().numpy()
        mask_resized = np.where(max_score_mask_resized != 0, 1, 0).astype(np.uint8)
        
        # 마스크 영역만 원본 이미지에서 추출
        masked_image = np.array(self.pf_image) * mask_resized[:, :, np.newaxis]
        
        # 투명한 배경을 추가하기 위해 RGBA로 변환
        alpha_channel = (mask_resized * 255).astype(np.uint8)  # 마스크가 적용된 부분에만 투명도 설정
        masked_image_rgba = np.dstack((masked_image, alpha_channel))
        
        # PIL 이미지로 변환하여 저장
        masked_image_pil = Image.fromarray(masked_image_rgba, 'RGBA')
        masked_image_pil.save('image.png')

    def run(self):
        inputs, outputs = self.process_image()
        masks, scores = self.post_process_masks(inputs, outputs)
        max_score_mask = self.extract_max_score_mask(masks, scores)
        self.save_mask_image(max_score_mask)
        self.save_masked_image(max_score_mask)


class ImageGridDisplay:
    def __init__(self, img1_path, img2_path, img3_path, img4_path, prompt):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.img3_path = img3_path
        self.img4_path = img4_path
        self.prompt = prompt
        self.images = []
        self.load_images()
    
    def load_images(self):
        # 이미지 경로에 있는 이미지를 읽고 RGB로 변환
        self.images.append(cv2.cvtColor(cv2.imread(self.img1_path), cv2.COLOR_BGR2RGB))
        self.images.append(cv2.cvtColor(cv2.imread(self.img2_path), cv2.COLOR_BGR2RGB))
        self.images.append(cv2.cvtColor(cv2.imread(self.img3_path), cv2.COLOR_BGR2RGB))
        self.images.append(cv2.cvtColor(cv2.imread(self.img4_path), cv2.COLOR_BGR2RGB))
        
    def display(self):
        # 2x2 서브플롯 설정
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        # 각 서브플롯에 이미지 출력
        titles = ['Original', 'Mask', 'Canny', 'Output']
        for i, ax in enumerate(axs.flat):
            ax.imshow(self.images[i])
            ax.set_title(titles[i])
            ax.axis('off')
        
        # 전체 타이틀 설정
        if self.prompt:
            plt.suptitle(self.prompt, fontsize=16)  # y 값으로 타이틀의 위치 조정 (기본값: 0.98)
    
            # 레이아웃 조정 및 출력
            plt.tight_layout()
            plt.show()