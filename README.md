# LTF End-to-End Inference 

This repo is the implementation of LTF test client for [HUGSIM benchmark](https://xdimlab.github.io/HUGSIM/)

The implementation is based on:
> [**NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking**](https://arxiv.org/abs/2406.15349)
> 
> [Daniel Dauner](https://danieldauner.github.io/)<sup>1,2</sup>, [Marcel Hallgarten](https://mh0797.github.io/)<sup>1,5</sup>, [Tianyu Li](https://github.com/sephyli)<sup>3</sup>, [Xinshuo Weng](https://xinshuoweng.com/)<sup>4</sup>, [Zhiyu Huang](https://mczhi.github.io/)<sup>4,6</sup>, [Zetong Yang](https://scholar.google.com/citations?user=oPiZSVYAAAAJ)<sup>3</sup>\
> [Hongyang Li](https://lihongyang.info/)<sup>3</sup>, [Igor Gilitschenski](https://www.gilitschenski.org/igor/)<sup>7,8</sup>, [Boris Ivanovic](https://www.borisivanovic.com/)<sup>4</sup>, [Marco Pavone](https://web.stanford.edu/~pavone/)<sup>4,9</sup>, [Andreas Geiger](https://www.cvlibs.net/)<sup>1,2</sup>, and [Kashyap Chitta](https://kashyap7x.github.io/)<sup>1,2</sup>  <br>
> 
> <sup>1</sup>University of Tübingen, <sup>2</sup>Tübingen AI Center, <sup>3</sup>OpenDriveLab at Shanghai AI Lab, <sup>4</sup>NVIDIA Research\
> <sup>5</sup>Robert Bosch GmbH, <sup>6</sup>Nanyang Technological University, <sup>7</sup>University of Toronto, <sup>8</sup>Vector Institute, <sup>9</sup>Stanford University
>
> Advances in Neural Information Processing Systems (NeurIPS), 2024 \
> Track on Datasets and Benchmarks 
<br/>

# Installation

You can use `pixi install` to easily install the environment of LTF.
The model weight of LTF can be downloaded from the official link: https://huggingface.co/autonomousvision/navsim_baselines/tree/main/ltf

Please change ${NAVSIM_PATH} in ltf_e2e.sh as the path on your machine.

# Launch Client

### Manually Launch
``` bash
zsh ./ltf_e2e.sh ${CUDA_ID} ${output_dir}
```

### Auto Lauch
The client can be auto lauched by the HUGSIM closed-loop script.
