# VR-Dex-Retargeting

VR enabled dexterous retargeting based on [AnyTeleop](https://yzqin.github.io/anyteleop/) project by Yuzhe Qin et al.

The `dex retargeting` module now supports all robot hand motion retargeting based on Meta Quest VR input. Quest App can be found in [this repo](https://github.com/wengmister/quest-wrist-tracker/tree/dex-retargeter).

## Changelog

<details>
<summary>Click to expand changelog</summary>

- Added VR UDP streaming data input for `dexpilot` retargeting algorithm.
- Added support for the following robotic hands
  - [Robotera XHand](https://www.robotera.com/en/goods1/4.html)

    ![xhand](/example/xhand.gif)

  - [Bidexhand](https://github.com/wengmister/BiDexHand)
  
    ![bidexhand](/example/bidexhand.gif)
  
  You will need to pull asset submodule from [this updated repo](https://github.com/wengmister/dex-urdf-plus)

</details>

## Demo

Navigate to `/example/vector_retargeting`

```python

python3 vr_realtime_retargeting.py --robot-name xhand --retargeting-type dexpilot --hand-type right

```

You will also need to `pip install` additional requirements under `example`

----------
Below are the original repo content.


<div align="center">
  <h1 align="center"> Dex Retargeting </h1>
  <h3 align="center">
    Various retargeting optimizers to translate human hand motion to robot hand motion.
  </h3>
</div>
<p align="center">
  <!-- code check badges -->
  <a href='https://github.com/dexsuite/dex-retargeting/blob/main/.github/workflows/test.yml'>
      <img src='https://github.com/dexsuite/dex-retargeting/actions/workflows/test.yml/badge.svg' alt='Test Status' />
  </a>
  <!-- issue badge -->
  <a href="https://github.com/dexsuite/dex-retargeting/issues">
  <img src="https://img.shields.io/github/issues-closed/dexsuite/dex-retargeting.svg" alt="Issues Closed">
  </a>
  <a href="https://github.com/dexsuite/dex-retargeting/issues?q=is%3Aissue+is%3Aclosed">
  <img src="https://img.shields.io/github/issues/dexsuite/dex-retargeting.svg" alt="Issues">
  </a>
  <!-- release badge -->
  <a href="https://github.com/dexsuite/dex-retargeting/tags">
  <img src="https://img.shields.io/github/v/release/dexsuite/dex-retargeting.svg?include_prereleases&sort=semver" alt="Releases">
  </a>
  <!-- pypi badge -->
  <a href="https://github.com/dexsuite/dex-retargeting/tags">
  <img src="https://static.pepy.tech/badge/dex_retargeting/month" alt="pypi">
  </a>
  <!-- license badge -->
  <a href="https://github.com/dexsuite/dex-retargeting/blob/main/LICENSE">
      <img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
  </a>
</p>
<div align="center">
  <h4>This repo originates from <a href="https://yzqin.github.io/anyteleop/">AnyTeleop Project</a></h4>
  <img src="example/vector_retargeting/teaser.webp" alt="Retargeting with different hands.">
</div>

## Installation

```shell
pip install dex_retargeting
```

To run the example, you may need additional dependencies for rendering and hand pose detection.

```shell
git clone https://github.com/dexsuite/dex-retargeting
cd dex-retargeting
pip install -e ".[example]"
```

## Changelog

### v0.5.0

- **Numpy Support Update**: Starting from this version, `dex-retargeting` supports `numpy >= 2.0.0`. If you need to use `numpy < 2.0.0`, you can install an earlier version of `dex-retargeting` using:
  ```bash
  pip install "dex-retargeting<0.5.0"
  ```

- **Mediapipe Compatibility**: Although `mediapipe` lists `numpy 1.x` as a dependency, it is compatible with `numpy >= 2.0.0`. You can safely ignore any warnings related to this and continue using `numpy 2.0.0` or higher.

- **Dependency Cleanup**: Removed `trimesh` as a dependency to simplify installation and reduce potential conflicts. The core functionality of `dex-retargeting` no longer requires mesh processing capabilities.

## Examples

### Retargeting from human hand video

This type of retargeting can be used for applications like teleoperation,
e.g. [AnyTeleop](https://yzqin.github.io/anyteleop/).

[Tutorial on retargeting from human hand video](example/vector_retargeting/README.md)

### Retarget from hand object pose dataset

![teaser](example/position_retargeting/hand_object.webp)

This type of retargeting can be used post-process human data for robot imitation,
e.g. [DexMV](https://yzqin.github.io/dexmv/).

[Tutorial on retargeting from hand-object pose dataset](example/position_retargeting/README.md)

## FAQ and Troubleshooting

### Joint Orders for Retargeting

URDF parsers, such as ROS, physical simulators, real robot driver, and this repository, may parse URDF files with
different joint orders. To use `dex-retargeting` results with other libraries, handle joint ordering explicitly **using
joint names**, which are unique within a URDF file.

Example: Using `dex-retargeting` with the SAPIEN simulator

```python
from dex_retargeting.seq_retarget import SeqRetargeting

retargeting: SeqRetargeting
sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
retargeting_joint_names = retargeting.joint_names
retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

# Use the index map to handle joint order differences
sapien_robot.set_qpos(retarget_qpos[retargeting_to_sapien])
```

This example retrieves joint names from the SAPIEN robot and `SeqRetargeting` object, creates a mapping
array (`retargeting_to_sapien`) to map joint indices, and sets the SAPIEN robot's joint positions using the retargeted
joint positions.

## Citation

This repository is derived from the [AnyTeleop Project](https://yzqin.github.io/anyteleop/) and is subject to ongoing
enhancements. If you utilize this work, please cite it as follows:

```shell
@inproceedings{qin2023anyteleop,
  title     = {AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System},
  author    = {Qin, Yuzhe and Yang, Wei and Huang, Binghao and Van Wyk, Karl and Su, Hao and Wang, Xiaolong and Chao, Yu-Wei and Fox, Dieter},
  booktitle = {Robotics: Science and Systems},
  year      = {2023}
}
```

## Acknowledgments

The robot hand models in this repository are sourced directly from [dex-urdf](https://github.com/dexsuite/dex-urdf).
The robot kinematics in this repo are based on [pinocchio](https://github.com/stack-of-tasks/pinocchio).
Examples use [SAPIEN](https://github.com/haosulab/SAPIEN) for rendering and visualization.

The `PositionOptimizer` leverages methodologies from our earlier
project, [From One Hand to Multiple Hands](https://yzqin.github.io/dex-teleop-imitation/).
Additionally, the `DexPilotOptimizer`is crafted using insights from [DexPilot](https://sites.google.com/view/dex-pilot).
