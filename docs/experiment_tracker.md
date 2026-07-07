# Experiment Tracker

Use this table to track planning, execution, results, and dissertation evidence. Status values should move from `Planned` to `Configured`, `Submitted`, `Completed`, `Analysed`, or `Skipped`.

## Run Tracking Table

| ID | Status | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Priority | GPUh | Result file | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E001 | Planned | Environment validation | All | Training | CUDA/PyTorch | 1 | N/A | Must Have | 0.5 | results/raw/E001.json |  |
| E002 | Planned | Environment validation | All | Training | PyTorch DDP/NCCL | 2 | N/A | Must Have | 0.5 | results/raw/E002.json |  |
| E003 | Planned | Environment validation | All | Training | PyTorch DDP/NCCL | 4 | N/A | Must Have | 1.0 | results/raw/E003.json |  |
| E004 | Planned | Environment validation | All | Training | FSDP | 2 | N/A | Must Have | 0.5 | results/raw/E004.json |  |
| E005 | Planned | Environment validation | All | Training | DeepSpeed ZeRO | 2 | N/A | Should Have | 0.5 | results/raw/E005.json |  |
| E006 | Planned | Environment validation | All | Inference | vLLM | 1 | N/A | Must Have | 0.5 | results/raw/E006.json |  |
| E007 | Planned | Environment validation | All | Inference | TensorRT-LLM | 1 | N/A | Optional | 0.8 | results/raw/E007.json |  |
| E008 | Planned | Baseline profiling | GPT-2 Small | Training | PyTorch single GPU | 1 | BF16 | Must Have | 1.0 | results/raw/E008.json |  |
| E009 | Planned | Baseline profiling | GPT-2 Small | Training | PyTorch single GPU repeat | 1 | BF16 | Must Have | 0.7 | results/raw/E009.json |  |
| E010 | Planned | Baseline profiling | GPT-2 Small | Inference | PyTorch eager | 1 | BF16 | Must Have | 0.5 | results/raw/E010.json |  |
| E011 | Planned | Baseline profiling | GPT-2 Small | Inference | PyTorch repeat | 1 | BF16 | Must Have | 0.4 | results/raw/E011.json |  |
| E012 | Planned | Baseline profiling | DistilBERT | Training | PyTorch single GPU | 1 | BF16 | Must Have | 1.0 | results/raw/E012.json |  |
| E013 | Planned | Baseline profiling | DistilBERT | Training | PyTorch single GPU repeat | 1 | BF16 | Must Have | 0.7 | results/raw/E013.json |  |
| E014 | Planned | Baseline profiling | DistilBERT | Inference | PyTorch eager | 1 | BF16 | Must Have | 0.5 | results/raw/E014.json |  |
| E015 | Planned | Baseline profiling | DistilBERT | Inference | PyTorch repeat | 1 | BF16 | Must Have | 0.4 | results/raw/E015.json |  |
| E016 | Planned | Baseline profiling | TinyLlama | Training | PyTorch single GPU | 1 | BF16 | Should Have | 1.5 | results/raw/E016.json |  |
| E017 | Planned | Baseline profiling | TinyLlama | Training | PyTorch single GPU repeat | 1 | BF16 | Should Have | 1.0 | results/raw/E017.json |  |
| E018 | Planned | Baseline profiling | TinyLlama | Inference | PyTorch eager | 1 | BF16 | Must Have | 0.5 | results/raw/E018.json |  |
| E019 | Planned | Baseline profiling | TinyLlama | Inference | PyTorch repeat | 1 | BF16 | Must Have | 0.4 | results/raw/E019.json |  |
| E020 | Planned | Workload characterisation | GPT-2 Small | Training | Profiler | 1 | BF16 | Must Have | 0.7 | results/raw/E020.json |  |
| E021 | Planned | Workload characterisation | GPT-2 Small | Training | Memory profiler | 1 | BF16 | Must Have | 0.7 | results/raw/E021.json |  |
| E022 | Planned | Workload characterisation | GPT-2 Small | Training | Data pipeline profiler | 1 | BF16 | Must Have | 0.5 | results/raw/E022.json |  |
| E023 | Planned | Workload characterisation | GPT-2 Small | Inference | vLLM/PyTorch profiler | 1 | BF16 | Should Have | 0.5 | results/raw/E023.json |  |
| E024 | Planned | Workload characterisation | DistilBERT | Training | Profiler | 1 | BF16 | Must Have | 0.7 | results/raw/E024.json |  |
| E025 | Planned | Workload characterisation | DistilBERT | Training | Memory profiler | 1 | BF16 | Must Have | 0.7 | results/raw/E025.json |  |
| E026 | Planned | Workload characterisation | DistilBERT | Training | Data pipeline profiler | 1 | BF16 | Must Have | 0.5 | results/raw/E026.json |  |
| E027 | Planned | Workload characterisation | DistilBERT | Inference | vLLM/PyTorch profiler | 1 | BF16 | Should Have | 0.5 | results/raw/E027.json |  |
| E028 | Planned | Workload characterisation | TinyLlama | Training | Profiler | 1 | BF16 | Must Have | 0.7 | results/raw/E028.json |  |
| E029 | Planned | Workload characterisation | TinyLlama | Training | Memory profiler | 1 | BF16 | Must Have | 0.7 | results/raw/E029.json |  |
| E030 | Planned | Workload characterisation | TinyLlama | Training | Data pipeline profiler | 1 | BF16 | Must Have | 0.5 | results/raw/E030.json |  |
| E031 | Planned | Workload characterisation | TinyLlama | Inference | vLLM/PyTorch profiler | 1 | BF16 | Should Have | 0.5 | results/raw/E031.json |  |
| E032 | Planned | Strong scaling | GPT-2 Small | Training | DDP | 1 | BF16 | Must Have | 0.8 | results/raw/E032.json |  |
| E033 | Planned | Strong scaling | GPT-2 Small | Training | DDP | 2 | BF16 | Must Have | 1.0 | results/raw/E033.json |  |
| E034 | Planned | Strong scaling | GPT-2 Small | Training | DDP | 4 | BF16 | Must Have | 2.0 | results/raw/E034.json |  |
| E038 | Planned | Strong scaling | DistilBERT | Training | DDP | 1 | BF16 | Must Have | 0.8 | results/raw/E038.json |  |
| E039 | Planned | Strong scaling | DistilBERT | Training | DDP | 2 | BF16 | Must Have | 1.0 | results/raw/E039.json |  |
| E040 | Planned | Strong scaling | DistilBERT | Training | DDP | 4 | BF16 | Must Have | 2.0 | results/raw/E040.json |  |
| E044 | Planned | Strong scaling | TinyLlama | Training | DDP or FSDP | 1 | BF16 | Optional | 1.0 | results/raw/E044.json |  |
| E045 | Planned | Strong scaling | TinyLlama | Training | DDP or FSDP | 2 | BF16 | Optional | 1.2 | results/raw/E045.json |  |
| E046 | Planned | Strong scaling | TinyLlama | Training | DDP or FSDP | 4 | BF16 | Optional | 2.4 | results/raw/E046.json |  |
| E035 | Planned | Weak scaling | GPT-2 Small | Training | DDP | 1 | BF16 | Must Have | 0.8 | results/raw/E035.json |  |
| E036 | Planned | Weak scaling | GPT-2 Small | Training | DDP | 2 | BF16 | Must Have | 1.0 | results/raw/E036.json |  |
| E037 | Planned | Weak scaling | GPT-2 Small | Training | DDP | 4 | BF16 | Must Have | 2.0 | results/raw/E037.json |  |
| E041 | Planned | Weak scaling | DistilBERT | Training | DDP | 1 | BF16 | Must Have | 0.8 | results/raw/E041.json |  |
| E042 | Planned | Weak scaling | DistilBERT | Training | DDP | 2 | BF16 | Must Have | 1.0 | results/raw/E042.json |  |
| E043 | Planned | Weak scaling | DistilBERT | Training | DDP | 4 | BF16 | Must Have | 2.0 | results/raw/E043.json |  |
| E047 | Planned | Distributed training strategy | GPT-2 Small | Training | DDP | 2 | BF16 | Must Have | 1.2 | results/raw/E047.json |  |
| E048 | Planned | Distributed training strategy | GPT-2 Small | Training | DDP | 4 | BF16 | Must Have | 2.4 | results/raw/E048.json |  |
| E049 | Planned | Distributed training strategy | GPT-2 Small | Training | FSDP full shard | 2 | BF16 | Must Have | 1.2 | results/raw/E049.json |  |
| E050 | Planned | Distributed training strategy | GPT-2 Small | Training | FSDP full shard | 4 | BF16 | Must Have | 2.4 | results/raw/E050.json |  |
| E051 | Planned | Distributed training strategy | GPT-2 Small | Training | FSDP sharded grad | 2 | BF16 | Should Have | 1.2 | results/raw/E051.json |  |
| E052 | Planned | Distributed training strategy | GPT-2 Small | Training | ZeRO-1 | 2 | BF16 | Should Have | 1.2 | results/raw/E052.json |  |
| E053 | Planned | Distributed training strategy | GPT-2 Small | Training | ZeRO-2 | 2 | BF16 | Should Have | 1.2 | results/raw/E053.json |  |
| E054 | Planned | Distributed training strategy | GPT-2 Small | Training | ZeRO-3 | 2 | BF16 | Optional | 1.2 | results/raw/E054.json |  |
| E055 | Planned | Distributed training strategy | DistilBERT | Training | DDP | 2 | BF16 | Must Have | 1.2 | results/raw/E055.json |  |
| E056 | Planned | Distributed training strategy | DistilBERT | Training | DDP | 4 | BF16 | Must Have | 2.4 | results/raw/E056.json |  |
| E057 | Planned | Distributed training strategy | DistilBERT | Training | FSDP full shard | 2 | BF16 | Must Have | 1.2 | results/raw/E057.json |  |
| E058 | Planned | Distributed training strategy | DistilBERT | Training | FSDP full shard | 4 | BF16 | Must Have | 2.4 | results/raw/E058.json |  |
| E059 | Planned | Distributed training strategy | DistilBERT | Training | FSDP sharded grad | 2 | BF16 | Should Have | 1.2 | results/raw/E059.json |  |
| E060 | Planned | Distributed training strategy | DistilBERT | Training | ZeRO-1 | 2 | BF16 | Should Have | 1.2 | results/raw/E060.json |  |
| E061 | Planned | Distributed training strategy | DistilBERT | Training | ZeRO-2 | 2 | BF16 | Should Have | 1.2 | results/raw/E061.json |  |
| E062 | Planned | Distributed training strategy | DistilBERT | Training | ZeRO-3 | 2 | BF16 | Optional | 1.2 | results/raw/E062.json |  |
| E063 | Planned | Memory optimisation | GPT-2 Small | Training | FSDP | 2 | BF16 | Must Have | 1.0 | results/raw/E063.json |  |
| E064 | Planned | Memory optimisation | GPT-2 Small | Training | FSDP | 2 | BF16 | Must Have | 1.0 | results/raw/E064.json |  |
| E065 | Planned | Memory optimisation | GPT-2 Small | Training | DDP | 1 | BF16 | Should Have | 0.6 | results/raw/E065.json |  |
| E066 | Planned | Memory optimisation | GPT-2 Small | Training | DDP | 1 | BF16 | Must Have | 0.6 | results/raw/E066.json |  |
| E067 | Planned | Memory optimisation | GPT-2 Small | Training | DDP | 1 | BF16 | Must Have | 0.6 | results/raw/E067.json |  |
| E068 | Planned | Memory optimisation | GPT-2 Small | Training | DDP | 1 | BF16 | Should Have | 0.6 | results/raw/E068.json |  |
| E069 | Planned | Memory optimisation | GPT-2 Small | Training | DDP | 2 | BF16 | Should Have | 0.7 | results/raw/E069.json |  |
| E070 | Planned | Memory optimisation | GPT-2 Small | Training | DDP | 2 | BF16 | Optional | 0.7 | results/raw/E070.json |  |
| E071 | Planned | Memory optimisation | GPT-2 Small | Training | DDP | 2 | BF16 | Should Have | 0.7 | results/raw/E071.json |  |
| E072 | Planned | Memory optimisation | GPT-2 Small | Training | DDP | 2 | BF16 | Optional | 0.7 | results/raw/E072.json |  |
| E073 | Planned | Memory optimisation | DistilBERT | Training | FSDP | 2 | BF16 | Must Have | 1.0 | results/raw/E073.json |  |
| E074 | Planned | Memory optimisation | DistilBERT | Training | FSDP | 2 | BF16 | Must Have | 1.0 | results/raw/E074.json |  |
| E075 | Planned | Memory optimisation | DistilBERT | Training | DDP | 1 | BF16 | Should Have | 0.6 | results/raw/E075.json |  |
| E076 | Planned | Memory optimisation | DistilBERT | Training | DDP | 1 | BF16 | Must Have | 0.6 | results/raw/E076.json |  |
| E077 | Planned | Memory optimisation | DistilBERT | Training | DDP | 1 | BF16 | Must Have | 0.6 | results/raw/E077.json |  |
| E078 | Planned | Memory optimisation | DistilBERT | Training | DDP | 1 | BF16 | Should Have | 0.6 | results/raw/E078.json |  |
| E079 | Planned | Memory optimisation | DistilBERT | Training | DDP | 2 | BF16 | Should Have | 0.7 | results/raw/E079.json |  |
| E080 | Planned | Memory optimisation | DistilBERT | Training | DDP | 2 | BF16 | Optional | 0.7 | results/raw/E080.json |  |
| E081 | Planned | Memory optimisation | DistilBERT | Training | DDP | 2 | BF16 | Should Have | 0.7 | results/raw/E081.json |  |
| E082 | Planned | Memory optimisation | DistilBERT | Training | DDP | 2 | BF16 | Optional | 0.7 | results/raw/E082.json |  |
| E083 | Planned | Precision study | GPT-2 Small | Training | PyTorch/Transformer Engine | 1 | FP32 | Must Have | 0.8 | results/raw/E083.json |  |
| E084 | Planned | Precision study | GPT-2 Small | Training | PyTorch/Transformer Engine | 1 | FP16 | Must Have | 0.8 | results/raw/E084.json |  |
| E085 | Planned | Precision study | GPT-2 Small | Training | PyTorch/Transformer Engine | 1 | BF16 | Must Have | 0.8 | results/raw/E085.json |  |
| E086 | Planned | Precision study | GPT-2 Small | Training | PyTorch/Transformer Engine | 1 | FP8 | Should Have | 0.8 | results/raw/E086.json |  |
| E087 | Planned | Precision study | GPT-2 Small | Inference | vLLM/PyTorch | 1 | FP16 | Must Have | 0.5 | results/raw/E087.json |  |
| E088 | Planned | Precision study | GPT-2 Small | Inference | vLLM/PyTorch | 1 | BF16 | Must Have | 0.5 | results/raw/E088.json |  |
| E089 | Planned | Precision study | GPT-2 Small | Inference | vLLM/PyTorch | 1 | FP8 | Should Have | 0.5 | results/raw/E089.json |  |
| E090 | Planned | Precision study | DistilBERT | Training | PyTorch/Transformer Engine | 1 | FP32 | Must Have | 0.8 | results/raw/E090.json |  |
| E091 | Planned | Precision study | DistilBERT | Training | PyTorch/Transformer Engine | 1 | FP16 | Must Have | 0.8 | results/raw/E091.json |  |
| E092 | Planned | Precision study | DistilBERT | Training | PyTorch/Transformer Engine | 1 | BF16 | Must Have | 0.8 | results/raw/E092.json |  |
| E093 | Planned | Precision study | DistilBERT | Training | PyTorch/Transformer Engine | 1 | FP8 | Should Have | 0.8 | results/raw/E093.json |  |
| E094 | Planned | Precision study | DistilBERT | Inference | vLLM/PyTorch | 1 | FP16 | Must Have | 0.5 | results/raw/E094.json |  |
| E095 | Planned | Precision study | DistilBERT | Inference | vLLM/PyTorch | 1 | BF16 | Must Have | 0.5 | results/raw/E095.json |  |
| E096 | Planned | Precision study | DistilBERT | Inference | vLLM/PyTorch | 1 | FP8 | Should Have | 0.5 | results/raw/E096.json |  |
| E097 | Planned | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | Should Have | 0.4 | results/raw/E097.json |  |
| E098 | Planned | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | Must Have | 0.4 | results/raw/E098.json |  |
| E099 | Planned | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | Must Have | 0.4 | results/raw/E099.json |  |
| E100 | Planned | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | Should Have | 0.4 | results/raw/E100.json |  |
| E101 | Planned | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | Should Have | 0.4 | results/raw/E101.json |  |
| E102 | Planned | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | Must Have | 0.4 | results/raw/E102.json |  |
| E103 | Planned | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | Should Have | 0.4 | results/raw/E103.json |  |
| E104 | Planned | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | Must Have | 0.4 | results/raw/E104.json |  |
| E105 | Planned | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | Should Have | 0.4 | results/raw/E105.json |  |
| E106 | Planned | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | Must Have | 0.4 | results/raw/E106.json |  |
| E107 | Planned | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | Must Have | 0.4 | results/raw/E107.json |  |
| E108 | Planned | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | Should Have | 0.4 | results/raw/E108.json |  |
| E109 | Planned | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | Should Have | 0.4 | results/raw/E109.json |  |
| E110 | Planned | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | Must Have | 0.4 | results/raw/E110.json |  |
| E111 | Planned | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | Should Have | 0.4 | results/raw/E111.json |  |
| E112 | Planned | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | Must Have | 0.4 | results/raw/E112.json |  |
| E113 | Planned | Communication analysis | GPT-2 Small | Training | NCCL/DDP | 2 | BF16 | Must Have | 0.8 | results/raw/E113.json |  |
| E114 | Planned | Communication analysis | GPT-2 Small | Training | FSDP/NCCL | 2 | BF16 | Must Have | 0.8 | results/raw/E114.json |  |
| E115 | Planned | Communication analysis | GPT-2 Small | Training | NCCL/DDP | 4 | BF16 | Must Have | 1.6 | results/raw/E115.json |  |
| E116 | Planned | Communication analysis | GPT-2 Small | Training | FSDP/NCCL | 4 | BF16 | Must Have | 1.6 | results/raw/E116.json |  |
| E117 | Planned | Communication analysis | DistilBERT | Training | NCCL/DDP | 2 | BF16 | Must Have | 0.8 | results/raw/E117.json |  |
| E118 | Planned | Communication analysis | DistilBERT | Training | FSDP/NCCL | 2 | BF16 | Must Have | 0.8 | results/raw/E118.json |  |
| E119 | Planned | Communication analysis | DistilBERT | Training | NCCL/DDP | 4 | BF16 | Must Have | 1.6 | results/raw/E119.json |  |
| E120 | Planned | Communication analysis | DistilBERT | Training | FSDP/NCCL | 4 | BF16 | Must Have | 1.6 | results/raw/E120.json |  |
| E121 | Planned | Hardware comparison | GPT-2 Small | Training | H100 | 1 | BF16 | Must Have | 0.8 | results/raw/E121.json |  |
| E122 | Planned | Hardware comparison | GPT-2 Small | Inference | H100 | 1 | BF16 | Should Have | 0.5 | results/raw/E122.json |  |
| E123 | Planned | Hardware comparison | GPT-2 Small | Training | L40S | 1 | BF16 | Should Have | 0.8 | results/raw/E123.json |  |
| E124 | Planned | Hardware comparison | GPT-2 Small | Inference | L40S | 1 | BF16 | Should Have | 0.5 | results/raw/E124.json |  |
| E125 | Planned | Hardware comparison | DistilBERT | Training | H100 | 1 | BF16 | Must Have | 0.8 | results/raw/E125.json |  |
| E126 | Planned | Hardware comparison | DistilBERT | Inference | H100 | 1 | BF16 | Should Have | 0.5 | results/raw/E126.json |  |
| E127 | Planned | Hardware comparison | DistilBERT | Training | L40S | 1 | BF16 | Should Have | 0.8 | results/raw/E127.json |  |
| E128 | Planned | Hardware comparison | DistilBERT | Inference | L40S | 1 | BF16 | Should Have | 0.5 | results/raw/E128.json |  |
| E129 | Planned | Inference framework | GPT-2 Small | Inference | PyTorch eager | 1 | BF16 | Must Have | 0.6 | results/raw/E129.json |  |
| E130 | Planned | Inference framework | GPT-2 Small | Inference | vLLM | 1 | BF16 | Must Have | 0.6 | results/raw/E130.json |  |
| E131 | Planned | Inference framework | GPT-2 Small | Inference | TensorRT-LLM | 1 | BF16 | Should Have | 0.6 | results/raw/E131.json |  |
| E153 | Planned | Inference framework | DistilBERT | Inference | PyTorch eager | 1 | BF16 | Must Have | 0.6 | results/raw/E153.json |  |
| E154 | Planned | Inference framework | DistilBERT | Inference | vLLM | 1 | BF16 | Must Have | 0.6 | results/raw/E154.json |  |
| E155 | Planned | Inference framework | DistilBERT | Inference | TensorRT-LLM | 1 | BF16 | Should Have | 0.6 | results/raw/E155.json |  |
| E177 | Planned | Inference framework | TinyLlama | Inference | PyTorch eager | 1 | BF16 | Must Have | 0.8 | results/raw/E177.json |  |
| E178 | Planned | Inference framework | TinyLlama | Inference | vLLM | 1 | BF16 | Must Have | 0.8 | results/raw/E178.json |  |
| E179 | Planned | Inference framework | TinyLlama | Inference | TensorRT-LLM | 1 | BF16 | Should Have | 0.8 | results/raw/E179.json |  |
| E132 | Planned | Inference batching | GPT-2 Small | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E132.json |  |
| E133 | Planned | Inference batching | GPT-2 Small | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E133.json |  |
| E134 | Planned | Inference batching | GPT-2 Small | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E134.json |  |
| E135 | Planned | Inference batching | GPT-2 Small | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E135.json |  |
| E136 | Planned | Inference batching | GPT-2 Small | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E136.json |  |
| E137 | Planned | Inference batching | GPT-2 Small | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E137.json |  |
| E156 | Planned | Inference batching | DistilBERT | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E156.json |  |
| E157 | Planned | Inference batching | DistilBERT | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E157.json |  |
| E158 | Planned | Inference batching | DistilBERT | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E158.json |  |
| E159 | Planned | Inference batching | DistilBERT | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E159.json |  |
| E160 | Planned | Inference batching | DistilBERT | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E160.json |  |
| E161 | Planned | Inference batching | DistilBERT | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E161.json |  |
| E180 | Planned | Inference batching | TinyLlama | Inference | vLLM | 1 | BF16 | Must Have | 0.5 | results/raw/E180.json |  |
| E181 | Planned | Inference batching | TinyLlama | Inference | vLLM | 1 | BF16 | Should Have | 0.5 | results/raw/E181.json |  |
| E182 | Planned | Inference batching | TinyLlama | Inference | vLLM | 1 | BF16 | Must Have | 0.5 | results/raw/E182.json |  |
| E183 | Planned | Inference batching | TinyLlama | Inference | vLLM | 1 | BF16 | Should Have | 0.5 | results/raw/E183.json |  |
| E184 | Planned | Inference batching | TinyLlama | Inference | vLLM | 1 | BF16 | Must Have | 0.5 | results/raw/E184.json |  |
| E185 | Planned | Inference batching | TinyLlama | Inference | vLLM | 1 | BF16 | Should Have | 0.5 | results/raw/E185.json |  |
| E138 | Planned | Inference concurrency | GPT-2 Small | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E138.json |  |
| E139 | Planned | Inference concurrency | GPT-2 Small | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E139.json |  |
| E140 | Planned | Inference concurrency | GPT-2 Small | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E140.json |  |
| E141 | Planned | Inference concurrency | GPT-2 Small | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E141.json |  |
| E142 | Planned | Inference concurrency | GPT-2 Small | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E142.json |  |
| E162 | Planned | Inference concurrency | DistilBERT | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E162.json |  |
| E163 | Planned | Inference concurrency | DistilBERT | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E163.json |  |
| E164 | Planned | Inference concurrency | DistilBERT | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E164.json |  |
| E165 | Planned | Inference concurrency | DistilBERT | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E165.json |  |
| E166 | Planned | Inference concurrency | DistilBERT | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E166.json |  |
| E186 | Planned | Inference concurrency | TinyLlama | Inference | vLLM | 1 | BF16 | Must Have | 0.5 | results/raw/E186.json |  |
| E187 | Planned | Inference concurrency | TinyLlama | Inference | vLLM | 1 | BF16 | Should Have | 0.5 | results/raw/E187.json |  |
| E188 | Planned | Inference concurrency | TinyLlama | Inference | vLLM | 1 | BF16 | Must Have | 0.5 | results/raw/E188.json |  |
| E189 | Planned | Inference concurrency | TinyLlama | Inference | vLLM | 1 | BF16 | Must Have | 0.5 | results/raw/E189.json |  |
| E190 | Planned | Inference concurrency | TinyLlama | Inference | vLLM | 1 | BF16 | Should Have | 0.5 | results/raw/E190.json |  |
| E143 | Planned | Inference prompt length | GPT-2 Small | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E143.json |  |
| E144 | Planned | Inference prompt length | GPT-2 Small | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E144.json |  |
| E145 | Planned | Inference prompt length | GPT-2 Small | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E145.json |  |
| E146 | Planned | Inference prompt length | GPT-2 Small | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E146.json |  |
| E167 | Planned | Inference prompt length | DistilBERT | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E167.json |  |
| E168 | Planned | Inference prompt length | DistilBERT | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E168.json |  |
| E169 | Planned | Inference prompt length | DistilBERT | Inference | vLLM | 1 | BF16 | Must Have | 0.4 | results/raw/E169.json |  |
| E170 | Planned | Inference prompt length | DistilBERT | Inference | vLLM | 1 | BF16 | Should Have | 0.4 | results/raw/E170.json |  |
| E191 | Planned | Inference prompt length | TinyLlama | Inference | vLLM | 1 | BF16 | Must Have | 0.5 | results/raw/E191.json |  |
| E192 | Planned | Inference prompt length | TinyLlama | Inference | vLLM | 1 | BF16 | Should Have | 0.5 | results/raw/E192.json |  |
| E193 | Planned | Inference prompt length | TinyLlama | Inference | vLLM | 1 | BF16 | Must Have | 0.5 | results/raw/E193.json |  |
| E194 | Planned | Inference prompt length | TinyLlama | Inference | vLLM | 1 | BF16 | Should Have | 0.5 | results/raw/E194.json |  |
| E147 | Planned | Inference multi-GPU | GPT-2 Small | Inference | vLLM tensor parallel | 2 | BF16 | Should Have | 0.8 | results/raw/E147.json |  |
| E148 | Planned | Inference multi-GPU | GPT-2 Small | Inference | vLLM tensor parallel | 4 | BF16 | Should Have | 1.6 | results/raw/E148.json |  |
| E171 | Planned | Inference multi-GPU | DistilBERT | Inference | vLLM tensor parallel | 2 | BF16 | Should Have | 0.8 | results/raw/E171.json |  |
| E172 | Planned | Inference multi-GPU | DistilBERT | Inference | vLLM tensor parallel | 4 | BF16 | Should Have | 1.6 | results/raw/E172.json |  |
| E195 | Planned | Inference multi-GPU | TinyLlama | Inference | vLLM tensor parallel | 2 | BF16 | Should Have | 0.8 | results/raw/E195.json |  |
| E196 | Planned | Inference multi-GPU | TinyLlama | Inference | vLLM tensor parallel | 4 | BF16 | Should Have | 1.6 | results/raw/E196.json |  |
| E149 | Planned | Inference precision/quantization | GPT-2 Small | Inference | vLLM or TensorRT-LLM | 1 | FP16 | Must Have | 0.5 | results/raw/E149.json |  |
| E150 | Planned | Inference precision/quantization | GPT-2 Small | Inference | vLLM or TensorRT-LLM | 1 | BF16 | Must Have | 0.5 | results/raw/E150.json |  |
| E151 | Planned | Inference precision/quantization | GPT-2 Small | Inference | vLLM or TensorRT-LLM | 1 | FP8 | Should Have | 0.5 | results/raw/E151.json |  |
| E152 | Planned | Inference precision/quantization | GPT-2 Small | Inference | vLLM or TensorRT-LLM | 1 | INT8/4-bit quantized | Optional | 0.5 | results/raw/E152.json |  |
| E173 | Planned | Inference precision/quantization | DistilBERT | Inference | vLLM or TensorRT-LLM | 1 | FP16 | Must Have | 0.5 | results/raw/E173.json |  |
| E174 | Planned | Inference precision/quantization | DistilBERT | Inference | vLLM or TensorRT-LLM | 1 | BF16 | Must Have | 0.5 | results/raw/E174.json |  |
| E175 | Planned | Inference precision/quantization | DistilBERT | Inference | vLLM or TensorRT-LLM | 1 | FP8 | Should Have | 0.5 | results/raw/E175.json |  |
| E176 | Planned | Inference precision/quantization | DistilBERT | Inference | vLLM or TensorRT-LLM | 1 | INT8/4-bit quantized | Optional | 0.5 | results/raw/E176.json |  |
| E197 | Planned | Inference precision/quantization | TinyLlama | Inference | vLLM or TensorRT-LLM | 1 | FP16 | Must Have | 0.5 | results/raw/E197.json |  |
| E198 | Planned | Inference precision/quantization | TinyLlama | Inference | vLLM or TensorRT-LLM | 1 | BF16 | Must Have | 0.5 | results/raw/E198.json |  |
| E199 | Planned | Inference precision/quantization | TinyLlama | Inference | vLLM or TensorRT-LLM | 1 | FP8 | Should Have | 0.5 | results/raw/E199.json |  |
| E200 | Planned | Inference precision/quantization | TinyLlama | Inference | vLLM or TensorRT-LLM | 1 | INT8/4-bit quantized | Optional | 0.5 | results/raw/E200.json |  |

## Analysis and Dashboard Tasks

| ID | Status | Analysis task | Purpose | Notes |
| --- | --- | --- | --- | --- |
| A01 | Planned | Training efficiency comparison | Identify best training setup by workload and GPU budget | Uses completed experiment results; no additional GPU time |
| A02 | Planned | Inference efficiency comparison | Identify best serving setup by latency and throughput target | Uses completed experiment results; no additional GPU time |
| A03 | Planned | Bottleneck classification | Classify each workload as compute, memory, communication, I/O, or mixed | Uses completed experiment results; no additional GPU time |
| A04 | Planned | GPU-hour comparison | Compare useful work per GPU-hour across configurations | Uses completed experiment results; no additional GPU time |
| A05 | Planned | Throughput/Watt comparison | Compare energy efficiency across selected runs | Uses completed experiment results; no additional GPU time |
| A06 | Planned | H100 vs L40S synthesis | Summarise hardware trade-offs and when H100 is justified | Uses completed experiment results; no additional GPU time |
| A07 | Planned | Dashboard build | Create visual dashboard for scaling, bottlenecks, cost and energy | Uses completed experiment results; no additional GPU time |
| A08 | Planned | Deployment recommendations | Translate findings into practical guidance | Uses completed experiment results; no additional GPU time |
