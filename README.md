<div align="center"><h1>&nbsp;Double: Breaking the Acceleration Limit via Double Retrieval Speculative Parallelism</h1></div>

<p align="center">
| <a href="https://arxiv.org/abs/2601.05524"><b>Paper (ACL 2026)</b></a> | 
<a href="双重检索并行投机采样 - 沐风rs的文章 - 知乎
https://zhuanlan.zhihu.com/p/1994067174113956612"><b>知乎</b>
https://zhuanlan.zhihu.com/p/1994067174113956612</a> |
</p>

*News* 🔥
- [2026/05] 🏆 **Double** is nominated as a **Best Paper Candidate** at ACL 2026!
- [2026/05] 🎉 **Double** has been accepted to **ACL 2026** as an **Oral Presentation**.
- [2026/04] We release the official implementation of Double, including the environment setup and datastore generation pipeline.

<br>

> **TL; DR:** We introduce **Double** (Double Retrieval Speculative Parallelism) to further reduce the inference latency of Large Language Models (LLMs) and break the current acceleration limits. Double is a **parallel** inference framework based on retrieval-based speculative decoding. It utilizes a dual-level retrieval mechanism (Global Datastore + Local Context) to fetch and verify drafts with unprecedented parallelism. 
>
> In summary, our Double is:
> - 🔥 **Blazing Fast:** Achieves highly significant speedups across major benchmarks (HumanEval, GSM8K, MT-bench).
> - 🛡️ **Provably Lossless:** The output distribution strictly matches standard auto-regressive decoding.
> - 🆓 **Training-Free:** Requires absolutely no draft-model training or parameter updates.

---

## 🛠️ Preparation & Installation

Follow the instructions below to prepare the environment for Double. We recommend using `conda` to manage your Python environment.

**1. Create and activate the environment:**
```bash
conda create -n double python=3.10 -y
conda activate double

```

**2. Clone the repository:**

```bash
git clone [https://github.com/Sylvan820/Double1.git](https://github.com/Sylvan820/Double1.git)
cd Double1

```

**3. Install dependencies:**
You can install the required packages using `pip` (or [uv](https://github.com/astral-sh/uv) for faster installation).

```bash
# Optional: pip install uv
pip install -r requirements.txt

```

*(Note: Ensure your `torch` version matches your CUDA environment. We recommend `torch>=2.1.0` for FlashAttention compatibility).*

---

## 🗄️ Datastore Generation

Since Double is a retrieval-based speculative decoding framework, it requires a pre-built **Global Datastore** extracted from a corpus to fetch potential token n-grams.

We provide automated scripts to tokenize your text corpus and build the highly-optimized retrieval index (e.g., Datastore Trie).

### Step 1: Prepare the Corpus

Prepare your raw text data in `.jsonl` format. For example, if you are evaluating on coding tasks, you might want to use a subset of the Stack dataset.

```json
// example data format (corpus.jsonl)
{"text": "def bubble_sort(arr): \n    n = len(arr)..."}
{"text": "import torch\nimport torch.nn as nn..."}

```

### Step 2: Build the Global Datastore

Run the `build_datastore.py` script. You need to specify the HuggingFace tokenizer of your target LLM so that the text is tokenized correctly before indexing.

```bash
python scripts/build_datastore.py \
    --corpus_path data/corpus.jsonl \
    --tokenizer_name meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir ./checkpoints/global_datastore \
    --max_chunk_size 2048 \
    --num_workers 8

```

**Arguments:**

* `--tokenizer_name`: The path or name of the target model's tokenizer.
* `--max_chunk_size`: Maximum sequence length for chunking the corpus.
* `--num_workers`: Number of CPU processes for parallel tokenization and index building.

Once finished, the datastore files (e.g., `index.bin`, `metadata.json`) will be saved in `./checkpoints/global_datastore`.

---

## 🚀 Quick Start & Reproduction

After setting up the environment and generating the datastore, you can directly run the evaluation scripts to reproduce the results in our paper.

### Run Parallel Speculative Decoding

Here is an example of evaluating Double on the HumanEval dataset using Llama-3-8B:

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch benchmark/eval_humaneval.py \
    --eval_mode double_retrieval \
    --target_model meta-llama/Meta-Llama-3-8B-Instruct \
    --datastore_path ./checkpoints/global_datastore \
    --max_tokens 1024 \
    --temp 0

```

### Compare with Baseline (Auto-Regressive)

To see the speedup, you can run the standard auto-regressive baseline by changing the `--eval_mode`:

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch benchmark/eval_humaneval.py \
    --eval_mode auto_regressive \
    --target_model meta-llama/Meta-Llama-3-8B-Instruct \
    --max_tokens 1024 \
    --temp 0

```

---

## 💡 FAQ

**1. Does Double support different model families?**
Yes. Double is entirely model-agnostic and training-free. As long as you build the datastore with the corresponding tokenizer, you can plug Double into Llama, Qwen, or Mistral architectures seamlessly.

**2. The datastore takes up too much memory. How can I optimize it?**
If memory or disk space is constrained, you can prune the datastore by setting a frequency threshold during building (`--min_frequency 2` in `build_datastore.py`), which filters out rare n-grams with negligible impact on the acceptance rate.

---

## 📖 Citation

If you find our work useful in your research, please consider citing our ACL 2026 paper:

```bibtex
@inproceedings{double2026acl,
    title={Double: Breaking the Acceleration Limit via Double Retrieval Speculative Parallelism},
    author={Sylvan820 and [Other Authors]},
    booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
    year={2026},
    url={[https://arxiv.org/abs/2601.05524](https://arxiv.org/abs/2601.05524)}
}

@misc{sylvan8202026doublebreakingaccelerationlimit,
      title={Double: Breaking the Acceleration Limit via Double Retrieval Speculative Parallelism}, 
      author={Sylvan820 and [Other Authors]},
      year={2026},
      eprint={2601.05524},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={[https://arxiv.org/abs/2601.05524](https://arxiv.org/abs/2601.05524)}, 
}

```

```

```
