<p align="center">
  <a href="https://pypi.org/project/pillow-dat/" target="_blank">
      <img alt="PIL × DAT" src="https://github.com/lovindata/pillow-dat/blob/main/.github/logo.png?raw=true" width="500" style="max-width: 100%;">
  </a>
</p>

<p align="center">
  PIL × DAT - Pillow extension for AI-based image upscaling.
</p>

<p align="center">
    <a href="https://github.com/lovindata/pillow-dat/blob/main/LICENSE"><img src="https://img.shields.io/github/license/lovindata/pillow-dat" alt="License"></a>
</p>

---

## Installation

For PyPI:

```bash
pip install pillow-dat
```

For Poetry:

```bash
poetry add pillow-dat
```

## Get started

```python
from PIL.Image import open

from PIL_DAT.Image import upscale

lumine_image = open(".github/lumine.png")
lumine_image = upscale(lumine_image, 2)
```

_Remark_: We strongly advocate for the utilization of `DAT light` models owing to their streamlined design and outstanding speed performance. However, should you opt for alternative models, please note that `*.pth` model weights can be accessed via [Google Drive](https://drive.google.com/drive/folders/1ro8bAZxrIEm03eE-7Lc15q9cwE3CJ-oh?usp=sharing).

## Example

|                                          Input (lumine.png)                                          |                                             DAT light (x2)                                              |                                              Bicubic (x2)                                              |
| :--------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| ![Input (lumine.png)](https://github.com/lovindata/pillow-dat/blob/main/.github/lumine.png?raw=true) | ![DAT light (x2)](https://github.com/lovindata/pillow-dat/blob/main/.github/lumine_output.png?raw=true) | ![Bicubic (x2)](https://github.com/lovindata/pillow-dat/blob/main/.github/lumine_bicubic.png?raw=true) |

## Benchmarks

### Speed

Performance benchmarks have been conducted on a computing system equipped with an `Intel(R) CORE(TM) i7-9750H CPU @ 2.60GHz processor`, accompanied by a `2 × 8 Go at 2667MHz RAM` configuration. Below are the recorded results:

|  _In seconds_  | 320 × 320 | 640 × 640 | 960 × 960 | 1280 × 1280 |
| :------------: | :-------: | :-------: | :-------: | :---------: |
| DAT light (x2) |   16.1    |   65.3    |   146.8   |    339.8    |
| DAT light (x3) |   14.3    |   61.7    |     -     |      -      |
| DAT light (x4) |   14.0    |   63.0    |     -     |      -      |

The results were compared against the renowned [`OpenCV`](https://opencv.org/) library, utilizing its `EDSR` model known for delivering superior image quality.

| _In seconds_ | 320 × 320 | 640 × 640 | 960 × 960 | 1280 × 1280 |
| :----------: | :-------: | :-------: | :-------: | :---------: |
|  EDSR (x2)   |   25.6    |   112.9   |   264.1   |    472.8    |
|  EDSR (x3)   |   24.3    |   112.5   |     -     |      -      |
|  EDSR (x4)   |   23.6    |   111.2   |     -     |      -      |

_Remark_: All benchmark results presented here are reproducible. For detailed implementation, please refer to the following files: [benchmark_speed_dat_light.py](https://github.com/lovindata/pillow-dat/blob/main/benchmarks/benchmark_speed_dat_light.py) and [benchmark_speed_edsr.py](https://github.com/lovindata/pillow-dat/blob/main/benchmarks/benchmark_speed_edsr.py).

### Quality

|                                             DAT light (x2)                                              |                                            EDSR (x2)                                             |
| :-----------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: |
| ![DAT light (x2)](https://github.com/lovindata/pillow-dat/blob/main/.github/lumine_output.png?raw=true) | ![EDSR (x2)](https://github.com/lovindata/pillow-dat/blob/main/.github/lumine_edsr.png?raw=true) |

_Remark_: All benchmark results presented here are reproducible. For detailed implementation, please refer to the following files: [example.py](https://github.com/lovindata/pillow-dat/blob/main/examples/example.py) and [benchmark_quality_edsr.py](https://github.com/lovindata/pillow-dat/blob/main/benchmarks/benchmark_quality_edsr.py).

### Alpha-channel-awareness

|                                         Input                                          |                                             DAT light (x2)                                             |                                            EDSR (x2)                                            |
| :------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------: |
| ![Input](https://github.com/lovindata/pillow-dat/blob/main/.github/apple.png?raw=true) | ![DAT light (x2)](https://github.com/lovindata/pillow-dat/blob/main/.github/apple_output.png?raw=true) | ![EDSR (x2)](https://github.com/lovindata/pillow-dat/blob/main/.github/apple_edsr.png?raw=true) |

_Remark_: All benchmark results presented here are reproducible. For detailed implementation, please refer to the following files: [benchmark_alpha_channel_awareness_dat_light.py](https://github.com/lovindata/pillow-dat/blob/main/benchmarks/benchmark_alpha_channel_awareness_dat_light.py) and [benchmark_alpha_channel_awareness_edsr.py](https://github.com/lovindata/pillow-dat/blob/main/benchmarks/benchmark_alpha_channel_awareness_edsr.py).

## Contribution

Please install [Python](https://www.python.org/downloads/).

Please install [Poetry](https://python-poetry.org/docs/#installation) via [pipx](https://pipx.pypa.io/stable/installation/).

Please install [VSCode](https://code.visualstudio.com/) and its extensions:

- Black Formatter
- isort
- Python
- Pylance
- Even Better TOML

To have your Python environment inside your project (optional):

```bash
poetry config virtualenvs.in-project true
```

To create your Python environment and install dependencies:

```bash
poetry install
```

To run unit tests:

```bash
pytest
```

To publish package:

```bash
poetry publish --build -u __token__ -p <pypi_token>
```

## Acknowledgement

This library is founded upon the pioneering research paper, ["Dual Aggregation Transformer for Image Super-Resolution"](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Dual_Aggregation_Transformer_for_Image_Super-Resolution_ICCV_2023_paper.pdf).

```
@inproceedings{chen2023dual,
    title={Dual Aggregation Transformer for Image Super-Resolution},
    author={Chen, Zheng and Zhang, Yulun and Gu, Jinjin and Kong, Linghe and Yang, Xiaokang and Yu, Fisher},
    booktitle={ICCV},
    year={2023}
}
```
