# SMPL Data Converter

AMASS 动作数据 → Drake / MuJoCo retargeting 格式。

## 准备

1. AMASS 数据（SMPL+H G 格式）放入 `input/`
2. Body model 放入 `models/
```bash   
pip install numpy scipy
```

## 使用

```bash
python convert_all.py
```

输出在 `output/<dataset>/Drake/` 和 `output/<dataset>/MuJoCo/`。

## 引用

```bibtex
@misc{AMASS_CMU,
    title = {{CMU MoCap Dataset}},
    author = {{Carnegie Mellon University}},
    url = {http://mocap.cs.cmu.edu}
}

@article{MANO:SIGGRAPHASIA:2017,
    title = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
    author = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
    journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
    volume = {36},
    number = {6},
    series = {245:1--245:17},
    month = nov,
    year = {2017},
    month_numeric = {11}
}
```
