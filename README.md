# GenRecv2
GenRecV2: Faster and Stable Interest Generation via Rectified Flow for Multimodal Recommendation


## Usage:


### DiffMM 实验， 对比模型

nohup python main.py -m DiffMM -d baby > ./output_DiffMM_baby_202600205_exp.log 2>&1 &

nohup python main.py -m DiffMM -d sports > ./output_DiffMM_sports_202600205_exp 2>&1 &

nohup python main.py -m DiffMM -d clothing > ./output_DiffMM_clothing_202600205_exp.log 2>&1 &

nohup python main.py -m DiffMM -d microlens > ./output_DiffMM_microlens_202600205_exp.log 2>&1 &

### GenRecv1实验， 对比模型

nohup python main.py -m GenRecV1 -d baby > ./output_GenRecV1_baby_202600205_exp.log 2>&1 &

nohup python main.py -m GenRecV1 -d sports > ./output_GenRecV1_sports_202600205_exp.log 2>&1 &

nohup python main.py -m GenRecV1 -d clothing > ./output_GenRecV1_clothing_202600205_exp.log 2>&1 &

nohup python main.py -m GenRecV1 -d microlens > ./output_GenRecV1_microlens_202600205_exp.log 2>&1 &


######GenRecv2  我们的模型#########

nohup python main.py -m GenRecv2 -d baby > ./output_GenRecv2_baby_202600205_exp_hypersearch_best2.log 2>&1 &

nohup python main.py -m GenRecv2 -d sports > ./output_GenRecv2_sports_202600205_exp_hypersearch_best2.log 2>&1 &

nohup python main.py -m GenRecv2 -d clothing > ./output_GenRecv2_clothing_202600205_exp_hypersearch_best2.log 2>&1 &

nohup python main.py -m GenRecv2 -d microlens > ./output_GenRecv2_microlens_202600205_exp_hypersearch_best2.log 2>&1 &