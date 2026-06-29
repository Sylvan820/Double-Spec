
<div align="center"><h1>&nbsp;Double: Breaking the Acceleration Limit via Double Retrieval Speculative Parallelism</h1></div>

<p align="center">
| <a href="#"><b>Paper (ACL 2026)</b></a> | 
<a href="#"><b>Blog</b></a> | <a href="#"><b>知乎</b></a> |
</p>

*News* 🔥
- [2026/05] 🏆 **Double** is nominated as a **Best Paper Candidate** at ACL 2026!
- [2026/05] 🎉 **Double** has been accepted to **ACL 2026** as an **Oral Presentation**.
- [2026/04] We release the official implementation of Double, including evaluation scripts for LLaMA-3 and Qwen-2.5 families. 

<center>
    <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://via.placeholder.com/800x400.png?text=Double:+Breaking+the+Acceleration+Limit" width="100%" alt="Double Speedup Overview"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
      Figure 1. Speedup on HumanEval and MT-Bench. All experiments are conducted with H100 80G GPUs. Double significantly surpasses traditional Speculative Decoding and single-retrieval methods.
    </div>
</center>

<br>

## Background & Prior Knowledge

The inference speed of Large Language Models (LLMs) is heavily bottlenecked by the memory bandwidth during auto-regressive decoding. To mitigate this, **Speculative Decoding (SD)** was introduced as a *draft-then-verify* paradigm: a smaller, faster model generates multiple draft tokens, which are then verified in parallel by the target LLM. 

However, training and maintaining a highly aligned draft model is expensive and cumbersome. Recently, **Retrieval-based Speculative Decoding** (e.g., REST) emerged, replacing the draft model with a non-parametric datastore (e.g., retrieving token n-grams). Yet, these methods often hit an "acceleration limit" due to low draft acceptance rates in out-of-domain scenarios or complex reasoning tasks.

## TL; DR: What is Double?

We introduce **Double** (Double Retrieval Speculative Parallelism) to break this acceleration limit. Double introduces a **dual-level retrieval mechanism**—fetching drafts simultaneously from both a *Global Datastore* (for general knowledge) and a *Local Context Cache* (for prompt-specific patterns). Combined with a novel speculative parallelism algorithm, Double pushes the boundaries of LLM inference latency.

In summary, **Double** is:
- 🔥 **Blazing Fast:** Achieves up to **4.2x**, **4.1x**, and **3.9x** speedup on HumanEval, GSM8K, and MT-Bench, respectively.
- 🛡️ **Provably Lossless:** The generated output distribution strictly matches the original target model.
- 🆓 **Training-Free & Model-Agnostic:** Requires zero additional parameter updates. Plug-and-play for any transformer-based LLM.
- 🧠 **Dual-Retrieval Parallelism:** Dynamically routes and verifies retrieved drafts from local and global memory in a highly parallelized tree structure.

---

<br>

<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Demo</h2>
    </div>
</div>

![AR-demo](https://via.placeholder.com/800x300.gif?text=Animation:+Auto-regressive+vs+Double)

<p align="center" style="color:gray;">Figure 2. Generation speed of Llama-3-70B using Double vs. standard auto-regressive decoding, running on an A100 GPU at bf16 precision. The highlighted tokens are parallelly accepted via Double's retrieval mechanism.</p>

---

<br>

<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Overview of Double</h2>
    </div>
</div>

Double consists of a Dual-Retrieval Engine (Global + Local) and a Speculative Parallelism Verifier. During inference, the engine retrieves matching token sequences from both datastores based on the current context. These retrieved sequences form a draft tree, which is then verified in a single forward pass by the target model. The engine adaptively decides the draft length based on the confidence of the dual-retrieval hit rate.

<center>
    <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://via.placeholder.com/800x400.png?text=Double+Architecture+Diagram" width="100%" alt="Double Architecture"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
      Figure 3. Overview of Double. It achieves unprecedented parallelism by combining Global and Local retrieval trees for the verification phase.
    </div>
</center>

## Preparation

Follow the instructions below to set up the environment and reproduce the results from the paper.

1. **Install dependencies**: We recommend using [uv](https://github.com/astral-sh/uv) for fast installation. Run `sh install.sh` to install all necessary packages.
2. **Activate environment**: After installation, run `source .venv/bin/activate`.
3. **Build the Datastore**: Run `python src/build_datastore.py --corpus <path_to_corpus>` to initialize the Global Datastore.
4. **Configure paths**: Update `src/config.yaml` with your target model paths, datastore paths, and evaluation datasets.

## Reproduction

We provide automated scripts for auto-regressive decoding, vanilla speculative decoding, single-retrieval baseline, and our Double retrieval framework. These scripts cover comparison benchmarks, ablation studies, and case studies.

```shell
# Run the core Double evaluation script
sh scripts/run_double_parallel.sh

```

## Examples

You can quickly test Double on the HumanEval benchmark using the following command:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 benchmark/eval_humaneval.py \
    --eval_mode double_retrieval \
    --target_model meta-llama/Meta-Llama-3-70B-Instruct \
    --datastore_path ./checkpoints/global_datastore \
    --max_tokens 1024 \
    --temp 0

```

## Interactive UI Demo

We provide a local web interface using Gradio to visualize how Double fetches and verifies tokens in real-time.

> Note: `app.py` is primarily for visualization purposes. When running the UI demo, make sure to enable the **"Use Double"** and **"Highlight Retrieved Tokens"** toggles to see the dual-retrieval mechanism in action.

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch app.py \
    --target_model meta-llama/Meta-Llama-3-8B-Instruct \
    --datastore_path ./checkpoints/global_datastore \
    --max_tokens 1024 

```

## FAQ

**1. The datastore takes up too much disk space. How can I reduce it?**
You can use our provided quantization script (`python src/compress_datastore.py`) to apply FP8 quantization to the retrieval embeddings, which reduces the disk footprint by ~4x with negligible impact on the acceptance rate.

**2. `RuntimeError: CUDA error: out of memory` during the parallel verification step.**
Double constructs a draft tree for verification. If your context window is extremely long, the parallel verification might OOM. Try reducing the `--max_tree_width` parameter in `src/config.yaml` (default is 64, reduce it to 32 or 16).

**3. Does Double support Qwen-2.5 and Llama-3 models?**
Yes! We have natively integrated custom attention kernels that support the precise rotational position embeddings (RoPE) used in both Qwen-2.5 and Llama-3 families. Double achieves excellent speedups on both architectures.

## Citation

If you find our work useful in your research, please consider citing our ACL 2026 paper:

```bibtex
@inproceedings{author2026double,
    title={Double: Breaking the Acceleration Limit via Double Retrieval Speculative Parallelism},
    author={Sylvan820 and [Other Authors]},
    booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
    year={2026},
    url={[https://arxiv.org/abs/xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)}
}

```

```

```
