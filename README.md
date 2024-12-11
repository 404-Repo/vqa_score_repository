## VQAScore for Evaluating Text-to-Visual Models 
**This is an optimized version of the original implementation that improves the inference of the default models.**

*Original implementation, paper, demo and dataset can be found via links below*

[[VQAScore Page](https://linzhiqiu.github.io/papers/vqascore/)] [[VQAScore Demo](https://huggingface.co/spaces/zhiqiulin/VQAScore)]  [[GenAI-Bench Page](https://linzhiqiu.github.io/papers/genai_bench/)] [[GenAI-Bench Demo](https://huggingface.co/spaces/BaiqiL/GenAI-Bench-DataViewer)] [[CLIP-FlanT5 Model Zoo](https://github.com/linzhiqiu/CLIP-FlanT5/blob/master/docs/MODEL_ZOO.md)]

### Publication by the authors of the method:

- **"VQAScore: Evaluating Text-to-Visual Generation with Image-to-Text Generation"** (ECCV 2024) [[Paper](https://arxiv.org/pdf/2404.01291)] [[HF](https://huggingface.co/zhiqiulin/clip-flant5-xxl)] <br>
[Zhiqiu Lin](https://linzhiqiu.github.io/), [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/), Baiqi Li, Jiayao Li, [Xide Xia](https://scholar.google.com/citations?user=FHLTntIAAAAJ&hl=en), [Graham Neubig](https://www.phontron.com/), [Pengchuan Zhang](https://scholar.google.com/citations?user=3VZ_E64AAAAJ&hl=en), [Deva Ramanan](https://www.cs.cmu.edu/~deva/)

- **"GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation"** (CVPR 2024, **Best Short Paper @ SynData Workshop**) [[Paper](https://arxiv.org/abs/2406.13743)] [[HF](https://huggingface.co/spaces/BaiqiL/GenAI-Bench-DataViewer)] <br>
Baiqi Li*, [Zhiqiu Lin*](https://linzhiqiu.github.io/), [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/), Jiayao Li, Yixin Fei, Kewen Wu, Tiffany Ling, [Xide Xia*](https://scholar.google.com/citations?user=FHLTntIAAAAJ&hl=en), [Pengchuan Zhang*](https://scholar.google.com/citations?user=3VZ_E64AAAAJ&hl=en), [Graham Neubig*](https://www.phontron.com/), [Deva Ramanan*](https://www.cs.cmu.edu/~deva/) (*Co-First and co-senior authors)

VQAScore significantly outperforms previous metrics such as CLIPScore and PickScore on compositional text prompts, and it is much simpler than prior art (e.g., ImageReward, HPSv2, TIFA, Davidsonian, VPEval, VIEScore) making use of human feedback or proprietary models like ChatGPT and GPT-4Vision. 

### Quick start

Install the package from the folder:
```bash
git clone https://github.com/linzhiqiu/t2v_metrics
cd t2v_metrics
pip install -e .
```
or via one line:
```bash
pip install git+https://github.com/404-Repo/vqa_score_repository.git@v1.0.0
```

Example of how to inference the model can be found in **vqa_score_tool.py**.

### Notes on GPU and cache
**GPU usage**: By default, this code uses the first cuda device on your machine. We recommend 40GB GPUs for the largest VQAScore models such as `clip-flant5-xxl` and `llava-v1.5-13b`. If you have limited GPU memory, consider smaller models such as `clip-flant5-xl` and `llava-v1.5-7b`.


### Customizing the question and answer template (for VQAScore)
The question and answer slightly affect the final score, as shown in the Appendix of the corresponding paper. 
We provide a simple default template for each model and do not recommend changing it for the sake of reproducibility. 
However, we do want to point out that the question and answer can be easily modified. 
For example, CLIP-FlanT5 and LLaVA-1.5 use the following template, 
which can be found at [t2v_metrics/clip_t5_model/clip_t5_model.py](t2v_metrics/clip_t5_model/clip_t5_model.py):

```python
# {} will be replaced by the caption
default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = 'Yes'
```

You can customize the template by passing the `question_template` and `answer_template` parameters into the `forward()` or `batch_forward()` functions:

```python
# Use a different question for VQAScore
scores = clip_flant5_score(images=images,
                           texts=texts,
                           question_template='Is this figure showing "{}"? Please answer yes or no.',
                           answer_template='Yes')
```

## Citation

If you find original work made by the authors of the paper useful, please use the following citation:

```
@article{lin2024evaluating,
  title={Evaluating Text-to-Visual Generation with Image-to-Text Generation},
  author={Lin, Zhiqiu and Pathak, Deepak and Li, Baiqi and Li, Jiayao and Xia, Xide and Neubig, Graham and Zhang, Pengchuan and Ramanan, Deva},
  journal={arXiv preprint arXiv:2404.01291},
  year={2024}
}
```
