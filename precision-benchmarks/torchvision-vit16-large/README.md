| Script                                                       | Model    | Matmul precision | Runtime   | Memory   | Train accuracy | Test accuracy |
| ------------------------------------------------------------ | -------- | ---------------- | --------- | -------- | -------------- | ------------- |
| [01_pytorch-fp32.py](http://01_pytorch-fp32.py)              | vit_l_16 | medium           | 16.88 min | 16.70 GB | 98.40%         | 94.06%        |
| [02_pytorch-fabric-fp32.py](http://02_pytorch-fabric-fp32.py) | vit_l_16 | medium           | 17.03 min | 16.70 GB | 98.49%         | 96.17%        |
| [03_fp16-mixed.py](http://03_fp16-mixed.py)                  | vit_l_16 | medium           | 11.63 min | 12.30 GB | 98.47%         | 94.79%        |
| [04_bf16-mixed.py](http://04_bf16-mixed.py/)                 | vit_l_16 | medium           | 11.31 min | 12.24 GB | 98.46%         | 95.62%        |
| [05_fp16-full.py](http://05_fp16-full.py)                    | vit_l_16 | medium           | 9.19 min  | 8.43 GB  | 10.02%         | 10.00%        |
| [06_bf16-full.py](http://06_bf16-full.py)                    | vit_l_16 | medium           | 9.37 min  | 8.43 GB  | 99.92%         | 97.86%        |

| PyTorch: 2.1.0+cu121 |
| -------------------- |
| Lightning: 2.1.0     |