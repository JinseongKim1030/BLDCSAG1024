# 파일 설명
main.ipynb    : 메인 실행 파일 \
bldc.py      : 필요한 클래스 정의 (BLD+ControlNet, SAM, ImageDisplay) \
img.png      : input image  \
original.png : resize된 img.png  \
mask.png     : SAM을 통과한 mask  \
image.png    : mask.png로 마스킹된 이미지 (BLD의 input으로 사용)  \
canny.png    : image.png  \
output.png   : 실행 결과 이미지  


# 사용 설명서
1. img.png의 이름으로 input image 저장 
2. main.ipynb 실행


# 모델 설명
1. img.png 넣으면 'SamImageProcessor' class가 'original.png', 'mask.png', 'image.png' 생성  
2. 생성된 'image.png'와 'mask.png'로 'BlendedLatentDiffusionWithControlNet' class가 'canny.png', 'output.png' 생성  
3. 'ImageGridDisplay' class는 쓰인 .png 파일들을 prompt와 함께 출력. (따로 저장은 안되니 필요하시면 복사해서 따로 저장해주세요.)  
