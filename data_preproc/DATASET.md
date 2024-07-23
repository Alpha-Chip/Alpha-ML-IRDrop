# Dataset for static IRDrop analysis 

1. Initial ICCAD 2023 contest data
* [Real circuits (10 testcases)](https://drive.google.com/file/d/18DFds-KuU4-yBo_TdaVqf1b31VCqI4hz/view)
* [Fake circuits (100 testcases)](https://drive.google.com/file/d/1s4ouZCIn6RxQ9XAAI00ygc1j0x_0Admx/view)

2. BeGAN generated data (2000 testcases).
* [Repository](https://github.com/UMN-EDA/BeGAN-benchmarks) Note: you need [nangate45 version](https://github.com/UMN-EDA/BeGAN-benchmarks/tree/master/BeGAN-circuit-benchmarks/nangate45)
* To convert to our training format:
`python data_preprpoc/convert_began_data.py`

3. Data generated from fake and real tests by changes in .sp files (Ilya).


4. Data generated for real circuits with OpenROAD (Evgeniy) 