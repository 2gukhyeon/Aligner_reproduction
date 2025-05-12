## Citation
```
@inproceedings{ji2024aligner,
  title={Aligner: Efficient Alignment by Learning to Correct},
  author={Jiaming Ji and Boyuan Chen and Hantao Lou and Donghai Hong and Borong Zhang and Xuehai Pan and Tianyi Qiu and Juntao Dai and Yaodong Yang},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=kq166jACVP}
}
```

# How to run
## Fine-tuning
```bash
python fine_tune.py --dataset_path "./dataset/aligner_train.jsonl"
```
**The train dataset is already included in the repository.**



## Installation
```
git clone https://github.com/2gukhyeon/Aligner_reproduction.git
conda create -n aligner python=3.10
conda activate aligner
cd Aligner_reproduction
pip install -r requirement.txt
```
