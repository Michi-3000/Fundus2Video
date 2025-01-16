# Fundus2Video

This repository contains the official implementation of our paper:  
**Fundus2Video: Cross-Modal Angiography Video Generation from Static Fundus Photography with Clinical Knowledge Guidance**  
Published at **MICCAI 2024**.  
You can find the paper [here](https://link.springer.com/chapter/10.1007/978-3-031-72378-0_64).

### Dependencies

This code is built upon the **pix2pixHD** framework developed by NVIDIA, which can be found at [pix2pixHD GitHub repository](https://github.com/NVIDIA/pix2pixHD).  

Install the required dependencies using:
```bash
pip install -r requirements.txt
```
### Dataset and Weights
Due to data privacy concerns, we are unable to provide the dataset and pre-trained weights used in this study.  

However, you can adapt the code to your own dataset by modifying the file:  
`data/aligned_dataset_fundus2video_tempo.py`  

Make sure your dataset follows a structure compatible with the expected format in the code.  

### Training
Run the following command to train the model:
```bash
python train.py
```


If you find this work useful, please cite our paper:
```bibtex
@inproceedings{zhang2024fundus2video,
  title={Fundus2Video: Cross-Modal Angiography Video Generation from Static Fundus Photography with Clinical Knowledge Guidance},
  author={Zhang, Weiyi and Huang, Siyu and Yang, Jiancheng and Chen, Ruoyu and Ge, Zongyuan and Zheng, Yingfeng and Shi, Danli and He, Mingguang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={689--699},
  year={2024},
  organization={Springer}
}
```
