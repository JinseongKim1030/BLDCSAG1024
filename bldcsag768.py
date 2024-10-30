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
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UNet2DConditionModel,
)
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

# CrossAttnStoreProcessor 클래스 정의 (SAG에 필요)
class CrossAttnStoreProcessor:
    def __init__(self):
        # 어텐션 확률을 저장할 변수 초기화
        self.attention_probs = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # 배치 크기와 시퀀스 길이 가져오기
        batch_size, sequence_length, _ = hidden_states.shape
        # 어텐션 마스크 준비
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # Query 생성
        query = attn.to_q(hidden_states)

        # Encoder Hidden States 설정
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # Key와 Value 생성
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Query, Key, Value를 헤드 차원으로 변환
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 어텐션 확률 계산 및 저장
        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # 어텐션 가중치와 Value를 곱하여 Hidden States 업데이트
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 선형 변환 및 드롭아웃 적용
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

# BlendedLatentDiffusionWithControlNet 클래스 정의
class BLDCSAG768:
    def __init__(self, prompt,negative_prompt, batch_size, blending_start_percentage, device):
        # 초기화: 입력된 파라미터들을 인스턴스 변수로 저장
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.init_image = 'image.png'
        self.mask = 'mask.png'
        self.model_path = 'stabilityai/stable-diffusion-2-1'
        self.controlnet_model_path = 'thibaud/controlnet-sd21-canny-diffusers'
        self.batch_size = batch_size
        self.blending_start_percentage = blending_start_percentage
        self.device = device
        self.output_path = 'output.png'
        # 모델 로드 메서드 호출
        self.load_models()

    def load_models(self):
        # UNet 모델 로드
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_path,
            subfolder="unet",
            torch_dtype=torch.float16,
            cross_attention_dim=1024  # cross_attention_dim 설정
        ).to(self.device)

        # ControlNet 모델 로드
        self.controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model_path,
            torch_dtype=torch.float16,
            cross_attention_dim=1024  # cross_attention_dim 설정
        ).to(self.device)

        # Stable Diffusion 2.1 파이프라인 로드
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_path,
            unet=self.unet,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
        ).to(self.device)

        # VAE, 토크나이저, 텍스트 인코더, 스케줄러 로드
        self.vae = self.pipe.vae.to(self.device)
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(self.model_path, subfolder="scheduler")

    @torch.no_grad()
    def edit_image(
        self,
        image_path,
        mask_path,
        prompts,
        negative_prompts,
        height=768,
        width=768,
        num_inference_steps=100,
        guidance_scale=7.0,
        generator=None,
        blending_percentage=0.10,
        sag_scale=0.8,  # SAG 스케일 추가
    ):
        # 랜덤 시드 설정
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(random.randint(1, 2147483647))

        # 1. 주어진 프롬프트의 개수를 기준으로 배치 크기 설정
        batch_size = len(prompts)

        # 2. 이미지 로드 및 리사이즈
        image = Image.open(image_path)
        image = image.resize((width, height), Image.BILINEAR)
        image = np.array(image)[:, :, :3]

        # 3. Canny Edge 이미지 생성
        canny_image = self._create_canny_image(image)
        Image.fromarray(canny_image).save('canny.png')
        controlnet_cond = self._prepare_control_image(canny_image)

        # 4. 원본 이미지를 latent space로 변환
        source_latents = self._image2latent(image)

        # 5. 마스크 로드 및 처리
        latent_mask, org_mask = self._read_mask(mask_path, dest_size=(height // 8, width // 8))

        # 6. 텍스트 임베딩 생성
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
        uncond_input = self.tokenizer(
            negative_prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # 초기 Latent 설정
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=torch.float16
        )

        # 타임스텝 설정
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 텐서 크기 맞추기
        source_latents = source_latents.repeat(batch_size, 1, 1, 1)
        latent_mask = latent_mask.repeat(batch_size, 1, 1, 1)
        controlnet_cond = controlnet_cond.repeat(batch_size, 1, 1, 1).to(self.device).half()

        # SAG를 위한 변수 초기화
        store_processor = CrossAttnStoreProcessor()
        original_attn_processors = self.unet.attn_processors
        map_size = None

        # 맵 사이즈를 얻기 위한 후크 함수
        def get_map_size(module, input, output):
            nonlocal map_size
            map_size = output[0].shape[-2:]

        # 어텐션 프로세서와 후크 등록
        self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor = store_processor
        self.unet.mid_block.attentions[0].register_forward_hook(get_map_size)

        # 타임스텝 루프 시작 (블렌딩 시작 시점 이후)
        for t in timesteps[int(len(timesteps) * blending_percentage):]:
            # Latent 모델 입력 스케일링
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # CFG를 위해 복제
            latent_model_input = torch.cat([latent_model_input] * 2, dim=0)
            controlnet_cond_in = torch.cat([controlnet_cond] * 2, dim=0)
            combined_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

            # ControlNet 적용
            controlnet_output = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=combined_embeddings,
                controlnet_cond=controlnet_cond_in,
                return_dict=False,
            )

            # UNet 적용
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=combined_embeddings,
                down_block_additional_residuals=controlnet_output[0],
                mid_block_additional_residual=controlnet_output[1],
            ).sample

            # Conditional & Unconditional 분리 및 CFG 적용
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # 어텐션 맵 저장
            uncond_attn, cond_attn = store_processor.attention_probs.chunk(2)
            uncond_attn = uncond_attn.detach()
            # 어텐션 확률 초기화
            store_processor.attention_probs = None

            # SAG 적용
            if sag_scale > 0.0:
                # x0와 epsilon 예측
                pred_x0 = self.pred_x0(latents, noise_pred_uncond, t)
                eps = self.pred_epsilon(latents, noise_pred_uncond, t)

                # SAG 마스킹
                degraded_latents = self.sag_masking(pred_x0, uncond_attn, map_size, t, eps)

                # Degraded 입력 준비
                degraded_latent_model_input = self.scheduler.scale_model_input(degraded_latents, t)

                # Degraded 입력에 대한 ControlNet 적용
                degraded_controlnet_output = self.controlnet(
                    degraded_latent_model_input,
                    t,
                    encoder_hidden_states=uncond_embeddings,
                    controlnet_cond=controlnet_cond,
                    return_dict=False,
                )

                # UNet 적용 (Unconditional embeddings 사용)
                degraded_noise_pred = self.unet(
                    degraded_latent_model_input,
                    t,
                    encoder_hidden_states=uncond_embeddings,
                    down_block_additional_residuals=degraded_controlnet_output[0],
                    mid_block_additional_residual=degraded_controlnet_output[1],
                ).sample

                # noise_pred 업데이트
                noise_pred += sag_scale * (noise_pred_uncond - degraded_noise_pred)

            # latents 업데이트
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # 마스크와 블렌딩
            noise_source_latents = self.scheduler.add_noise(
                source_latents,
                torch.randn(latents.shape, device=latents.device, generator=generator, dtype=latents.dtype),
                t
            )
            kernel = np.ones((1, 1), np.uint8)

            eroded_masks = torch.empty_like(latent_mask)

            # 각 채널에 대해 침식 수행
            for i in range(latent_mask.size(1)):
                mask_np = latent_mask[0, i].cpu().numpy()
                mask_np_uint8 = (mask_np * 255).astype(np.uint8)
                eroded_np_uint8 = cv2.dilate(mask_np_uint8, kernel, iterations=1)
                eroded_np = eroded_np_uint8.astype(np.float32) / 255.0
                eroded_masks[0, i] = torch.from_numpy(eroded_np).to(latent_mask.dtype)

            # Latents에 마스크 적용
            latents = latents * eroded_masks + noise_source_latents * (1 - latent_mask)

        # 원래의 어텐션 프로세서로 복원
        self.unet.set_attn_processor(original_attn_processors)

        # Latents를 이미지로 디코딩
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        # 후처리 및 반환
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images

    @torch.no_grad()
    def _image2latent(self, image):
        # 이미지를 텐서로 변환하고 정규화
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        image = image.half()
        # VAE를 통해 Latent로 인코딩
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215

        return latents

    def _read_mask(self, mask_path: str, dest_size=(96, 96)):
        # 마스크 이미지 로드 및 이진화
        org_mask = Image.open(mask_path).convert("L")
        mask = org_mask.resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask[np.newaxis, np.newaxis, ...]
        mask = torch.from_numpy(mask).half().to(self.device)

        return mask, org_mask

    def _create_canny_image(self, image):
        # OpenCV를 사용하여 Canny Edge 이미지 생성
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        edges = cv2.resize(edges, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return edges

    def _prepare_control_image(self, image):
        # 이미지를 텐서로 변환 및 전처리
        if not isinstance(image, torch.Tensor):
            image = np.array(image)
            if len(image.shape) == 2:
                image = image[:, :, None]
            image = image.transpose(2, 0, 1)  # (C, H, W)
            if image.shape[0] == 1:
                image = np.repeat(image, repeats=3, axis=0)
            image = image / 255.0
            image = torch.from_numpy(image).float()
        return image.to(self.device).half()

    # pred_x0 함수 정의
    def pred_x0(self, sample, model_output, timestep):
        # 알파 및 베타 값 계산
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].to(sample.device)
        beta_prod_t = 1 - alpha_prod_t
        # 예측 타입에 따라 pred_original_sample 계산
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        return pred_original_sample

    # pred_epsilon 함수 정의
    def pred_epsilon(self, sample, model_output, timestep):
        # 알파 및 베타 값 계산
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].to(sample.device)
        beta_prod_t = 1 - alpha_prod_t
        # 예측 타입에 따라 pred_eps 계산
        if self.scheduler.config.prediction_type == "epsilon":
            pred_eps = model_output
        elif self.scheduler.config.prediction_type == "sample":
            pred_eps = (sample - alpha_prod_t ** 0.5 * model_output) / beta_prod_t ** 0.5
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_eps = beta_prod_t ** 0.5 * sample + alpha_prod_t ** 0.5 * model_output
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        return pred_eps

    # sag_masking 함수 정의
    def sag_masking(self, original_latents, attn_map, map_size, t, eps):
        # 어텐션 맵의 크기 및 Latent 크기 가져오기
        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = self.unet.config.attention_head_dim
        if isinstance(h, list):
            h = h[-1]

        # 어텐션 마스크 생성
        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_mask = attn_map.mean(1).sum(1) > 1.0
        attn_mask = attn_mask.reshape(b, map_size[0], map_size[1]).unsqueeze(1).repeat(1, latent_channel, 1, 1).type(attn_map.dtype)
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))

        # 블러 적용
        degraded_latents = self.gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
        degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)

        # 노이즈 추가
        degraded_latents = self.scheduler.add_noise(degraded_latents, noise=eps, timesteps=t[None])

        return degraded_latents

    # gaussian_blur_2d 함수 정의
    def gaussian_blur_2d(self, img, kernel_size, sigma):
        # 가우시안 커널 생성
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        x_kernel = pdf / pdf.sum()
        x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)
        kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
        kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])
        padding = [kernel_size // 2] * 4
        # 이미지에 패딩 적용 및 컨볼루션
        img = F.pad(img, padding, mode="reflect")
        img = F.conv2d(img, kernel2d, groups=img.shape[-3])
        return img

# SamImageProcessor 클래스 정의
class SamImageProcessor:
    def __init__(self, img_path, mask_num, device, model_name="facebook/sam-vit-huge"):
        self.img_path = img_path
        self.device = device
        # 모델 및 프로세서 로드
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_name)
        # 이미지 로드
        self.pf_image = self._load_image()
        self.mask_num = mask_num % 3  # 마스크 번호 설정


    def _load_image(self):
        # 이미지 로드 및 리사이즈
        pf_image = Image.open(self.img_path).convert("RGB").resize((768, 768), Image.LANCZOS)
        pf_image.save('original.png')
        return pf_image

    def process_image(self):
        # 입력 포인트 설정 (이미지 중앙)
        input_points = [[[self.pf_image.size[0] // 2, self.pf_image.size[1] // 2]]]
        # 입력 데이터 생성
        inputs = self.processor(self.pf_image, input_points=input_points, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # 모델 추론
            outputs = self.model(**inputs)
        return inputs, outputs

    def post_process_masks(self, inputs, outputs):
        # 마스크 후처리
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores
        return masks, scores

    def extract_max_score_mask(self, masks, scores):
        # 최대 점수의 마스크 추출
        max_score_mask = masks[0][:, self.mask_num, :, :]
        return max_score_mask

    def save_mask_image(self, max_score_mask):
        # 마스크 이미지를 저장
        color = np.array([0, 0, 0, 0.9])
        h, w = max_score_mask.shape[-2:]
        mask_image = np.where(max_score_mask.reshape(h, w, 1) != 0, color.reshape(1, 1, -1), [1, 1, 1, 1])
        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8), 'RGBA')
        mask_image_pil.save('mask.png')

    def save_masked_image(self, max_score_mask):
        # 마스크된 이미지를 저장
        max_score_mask_resized = max_score_mask.squeeze().numpy()
        mask_resized = np.where(max_score_mask_resized != 0, 1, 0).astype(np.uint8)

        masked_image = np.array(self.pf_image) * mask_resized[:, :, np.newaxis]

        alpha_channel = (mask_resized * 255).astype(np.uint8)
        masked_image_rgba = np.dstack((masked_image, alpha_channel))

        masked_image_pil = Image.fromarray(masked_image_rgba, 'RGBA')
        masked_image_pil.save('image.png')

    def run(self):
        # 전체 프로세스 실행
        inputs, outputs = self.process_image()
        masks, scores = self.post_process_masks(inputs, outputs)
        max_score_mask = self.extract_max_score_mask(masks, scores)
        self.save_mask_image(max_score_mask)
        self.save_masked_image(max_score_mask)

# ImageGridDisplay 클래스 정의
class ImageGridDisplay:
    def __init__(self, img1_path, img2_path, img3_path, img4_path, prompt):
        # 이미지 경로 및 프롬프트 저장
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.img3_path = img3_path
        self.img4_path = img4_path
        self.prompt = prompt
        self.images = []
        # 이미지 로드 메서드 호출
        self.load_images()

    def load_images(self):
        # 이미지를 OpenCV로 로드하고 RGB로 변환하여 저장
        self.images.append(cv2.cvtColor(cv2.imread(self.img1_path), cv2.COLOR_BGR2RGB))
        self.images.append(cv2.cvtColor(cv2.imread(self.img2_path), cv2.COLOR_BGR2RGB))
        self.images.append(cv2.cvtColor(cv2.imread(self.img3_path), cv2.COLOR_BGR2RGB))
        self.images.append(cv2.cvtColor(cv2.imread(self.img4_path), cv2.COLOR_BGR2RGB))

    def display(self):
        # 이미지를 2x2 그리드로 디스플레이
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        titles = ['Original', 'Mask', 'Canny', 'Output']
        for i, ax in enumerate(axs.flat):
            ax.imshow(self.images[i])
            ax.set_title(titles[i])
            ax.axis('off')

        plt.tight_layout()
        plt.show()
