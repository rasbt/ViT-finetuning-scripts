| Script                                                       | Model    | Matmul precision | Runtime  | Memory  | Train accuracy | Test accuracy |
| ------------------------------------------------------------ | -------- | ---------------- | -------- | ------- | -------------- | ------------- |
| [01_pytorch-fp32.py](http://01_pytorch-fp32.py)              | vit_b_16 | medium           | 7.71 min | 3.71 GB | 97.96%         | 95.27%        |
| [02_pytorch-fabric-fp32.py](http://02_pytorch-fabric-fp32.py) | vit_b_16 | medium           | 7.53 min | 3.71 GB | 97.87%         | 95.54%        |
| [03_fp16-mixed.py](http://03_fp16-mixed.py)                  | vit_b_16 | medium           | 9.38 min | 3.03 GB | 97.94%         | 96.09%        |
| [04_bf16-mixed.py](http://04_bf16-mixed.py/)                 | vit_b_16 | medium           | 8.50 min | 3.03 GB | 97.86%         | 95.16%        |
| [05_fp16-full.py](http://05_fp16-full.py)                    | vit_b_16 | medium           | 7.22 min | 1.94 GB | 10.01%         | 10.00%        |
| [06_bf16-full.py](http://06_bf16-full.py)                    | vit_b_16 | medium           | 7.00 min | 1.95 GB | 99.69%         | 97.52%        |

| PyTorch: 2.1.0+cu121 |
| -------------------- |
| Lightning: 2.1.0     |