# DR-IQA

The paper is under review.

# Abstract

The emphasis on no-reference image quality assessment has often overshadowed the significance of Full-Reference Image Quality Assessment (FR-IQA), which generally better reflects human contrastive perception mechanism. However, FR-IQA presents challenges in obtaining content-aligned reference images. To tackle these issues, a novel Bidirectional Reference Image Quality Assessment (BRIQA) method is proposed, centering on leveraging bidirectional reference images and content-quality correlation modeling. First, triplets of content-aligned low-quality and content-non-aligned high-quality reference images are generated using two easily accessible approaches. To prevent the extraction of redundant information, two feature extractors pretrained through unsupervised contrastive learning are utilized to independently extract content and quality features for the triplet images. Then, an attention-mixer is introduced to further mine quality difference information and enhance content feature. Finally, a content-quality correlation modeler is proposed to model the relationship between quality differences and visual contents. Experimental results on benchmark datasets demonstrate that the BRIQA  outperforms existing state-of-the-art methods.

# Requirement

- basicsr
- datasets
- einops
- faiss
- learning
- matplotlib
- numpy
- openpyxl
- pandas
- Pillow
- rfconv
- scikit_learn
- scipy
- skimage
- timm
- torch
- torchvision

More information please check the requirements.txt.