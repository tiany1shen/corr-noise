# 舞蹈生成

## 1. 原始数据的分割和处理

<details>

TODO List

- [x] 获取分割列表
- [x] 分割保存并初步处理原始数据文件
  - [x] 处理原始动作文件
  - [x] 处理原始音频文件
- [x] 切分数据片段
  - [x] 提取音频特征
  - [x] 同步切分片段

### 1.1 数据集分割

数据集的分割依据保存在数据集目录下的 `splits/` 文件夹中，包含 `splits/train_list.txt`, `splits/test_list.txt` 和 `splits/ignore_list.txt` 三个文件。获取数据集的某个分割列表时，使用 `data/code/filter_split_data.py` 中的 `split_data_generator` 函数，返回一个生成器对象，其元素是该分割下包含的数据信息的三元组：(Sequence name, Motion file, Wav file)

### 1.2 处理动作数据

将原始动作数据统一处理为形状为 (Sequence length, 3 + number of joint * 3) 的 numpy.ndarray 对象，并保存为分割对应新目录下的 `.npy` 文件。以训练集为例，新目录为 `train/motion/*.npy` 。处理后的数据维度解释：前3个维度为人体骨骼根节点的位移（xyz坐标表示），后面维度为每个人体关节点的相对旋转（axis-angle表示）。

对于 EDGE 数据（实际上为 Aist++ 数据），原始动作数据为 `.pkl` 文件，其中保存的是一个字典，用来表示 24 个人体关节点的运动序列，其键对应的值为：

- smpl_loss (float): 与本项目无关；
- smpl_poses (ndarray): 24 个人体关节点的旋转，形状为 (length, 72)；
- smpl_scaling (ndarray): 标量，表示位移路径的比例系数；
- smpl_trans (ndarray): 位移路径，形状为 (length, 3).

处理 EDGE 原始动作数据的方式为：将路径缩放到基本比例，再和人体姿态在维度上拼接，最终得到形状为 (length, 75) 的 ndarray 数据。（从 60 fps 下采样到 30 fps）

对于 FineDance 数据，原始动作数据为 `.npy` 文件，其中保存的是一个 numpy.ndarray，形状为 (length, 315)，维度 315 表示：前 3 个维度为位移路径，后 312 维为 52 个关节点的 6d 旋转表示。处理 FineDance 数据的方式为：将 6d 旋转表示变换为 axis-angle 表示，再和位移路径拼接起来，最终得到形状为 (length, 159) 的 ndarray 数据。

### 1.3 处理音频文件

音频文件将直接被复制到新目录 `train/music/*.wav` 下保存。

### 1.4 提取音频特征

使用 Librosa 提取音频特征，保存到 `train/music/*.npy` 路径。

### 1.5 同步切分片段

对同名称的动作和音频数据进行分段，切分后的片段保存到 `train/slice/*/*_slice_001.npz`。默认条件下，片段长度为 5 秒，保存的文件中可以读取两个关键字：`motion` 和 `music`， 均为长度为 150 的 numpy.ndarray：对于 EDGE 数据，`motion` 的形状为 (150, 75)；对于 FineDance 数据，`motion` 的形状为 (150, 159) 。

在切分片段的过程中，计算动作是否有效：当 `clip.std(axis=0).mean() <= 0.07` 时，认为该片段是一个静止的片段，将其弃用至 `slice_discard/` 文件夹并跳过。

经过上述处理，数据目录构成如下：

```dir-structure
- (data root dir)
|-  (raw data dir)
|-  splits/train_list.txt;test_list.txt;ignore_list.txt
|-  train/
|   |-  motion/ *.npy
|   |-  wav/    *.wav
|   |-  music/  *.npy (music feature)
|   |-  slice/
|   |   |-  */  (sequence name)
|   |       |-  *_slice_00*.npz (sliced data, keys: ['motion', 'music'])
|   |-  slice_discard/
|       |- *_slice_00*.npz (discarded data clip)
|-  test/
```

</details>

## 2. 数据集与数据可视化

<details>

TODO List

- [x] sliced motion-music pair 数据集
  - [x] 在根目录下获取数据列表并读取数据
  - [x] 根节点位移预处理：减去起点坐标
  - [x] 调整 EDGE / FineDance 数据的坐标顺序
  - [x] 动作标准化：起始平面朝向
  - [x] 相对旋转预处理：变换为 6d 表示
- [x] 可视化判断数据的人体面向方向，
  - [x] 选择 SMPL 或 SMPLX 模型进行前向运动学计算
  - [x] 检查 T-pose，对不同关节数自适应绘图
  - [x] 渲染人体骨骼
  - [x] 将动作数据变换为 z-axis up（旋转根节点）

### 2.1 获取数据

经过 [原始数据处理](#1-原始数据的分割和处理) 得到的目录架构如下：

```dir-structure
/
|-  motion/ *.npy
|-  wav/    *.wav
|-  music/  *.npy 
|-  slice_discard/
|-  slice/
    |-  */  
        |-  *_slice_00*.npz
```

对于训练使用的固定序列长度的 sliced motion-music pair 数据，只需要关注 `slice/` 子目录下已经切分好的动作序列和音乐特征序列：

- `slice/` 目录下的每个文件夹对应一条完整的舞蹈动作序列，每个文件夹下的数据文件为时移步长为 2.5 秒、长度为 5 秒的切分片段，其命名格式为 `*_slice_{id}.npz`，id编号 从 0 开始；
- 片段的起始、结束时间点分别为 id \* 2.5、id \* 2.5 + 5 (second)。

### 2.2 SMPL-X 表示

**关节点使用**：

动作数据的人体关节点采用的是 SMPL-X 格式，但我们只关注其部分关节点的位置信息，只需要利用这些信息执行人体骨骼的前向运动学计算。

- EDGE 数据使用人体姿态的 24 个关节点（SMPL 格式），每个关节点的姿态由一个 3 维的 axis-angle 向量表示；
- FineDance 数据使用人体姿态的 52 个关节点（SMPL 中的左右手各自被 15 个手部关节点进行细节建模，每根手指 3 个关节）

进行骨骼计算时，使用 [SMPL-X_NEUTRAL](https://github.com/li-ronghui/FineDance/blob/main/smplx_neu_J_1.npy) 模型。该模型包含人体姿态(22)、面部姿态(3)和手部姿态(30)共 55 个关节：在单独建模动作时，使用 22 个人体关节和 2 个中指根部关节（作为左右手关节）模拟 SMPL 格式；在建模手部姿态时，使用人体姿态和手部姿态；由于使用的数据中不包含面部姿态，因此不予考虑。

**初始姿态调整**:

模型给出的初始姿态（人体骨骼 T-Pose）在世界坐标系下与常用视角和运动方向不一致，对齐进行一定的调整：

1. 调整 T-Pose 位置，使根关节点（骨盆位置）位于世界坐标系原点。具体的，全部关键点坐标减去根节点坐标；
2. 调整 T-Pose 坐标系，使头顶方向为 Z+，面向方向为 X+，原始为 (Y+, Z+)。具体的，将 XYZ 坐标顺序按照 (2, 0, 1) 进行交换；
3. （可选）添加法向量节点，用来指示人体正面方向。具体作用见 [关节点计算](#23-关节点计算)。

### 2.3 关节点计算

计算关节点坐标时，使用 `visualize.skeleton.SMPLX_Skeleton` 类的 `forward` 方法进行计算。需要注意的是，原始数据（即 `slice/` 文件夹下储存的数据）也与世界坐标系不一致，需要经过类似的预处理：1. 根节点路径标准化，使每个片段第一帧的根节点位置是原点；2. 调整姿态坐标系，将旋转的轴角表示的坐标顺序按照 (2, 0, 1) 进行交换。

```Python
joints, normal = SMPLX_Skeleton(with_normal=True).forward(rotations, root_positions)
```

`forward` 方法支持批量运算，并返回两项：动作序列的关节点坐标，人体姿态的法向量坐标。

**动作后处理**:

我们注意到，由于片段是均匀切割得到的，动作的初始朝向并不固定。此时我们使用计算得到的法向量将初始法向量朝向调整至 +XOZ 平面内。具体的，计算一个绕 Z+ 轴的旋转使第一帧的法向量旋转至 Y 分量为 0，并将这个旋转应用到整个运动序列每一帧的根节点上。这表示我们将世界坐标系绕 Z+ 轴做了一个旋转，从而使运动的第一帧面向 X+ 方向，由于 SMPL-X 格式除根节点外的旋转都是相对与父节点的，因此其他节点无需改动；对应的，需要对运动轨迹应用同样的旋转。

同样注意到，动作的初始姿态并不固定，这表示每个动作的地面高度不一致。因此我们计算每个动作的地面高度（首先计算每一帧最低的两个 Z 坐标的平均值，再计算这些平均值中最小的一半的平均值作为地面高度），并将这个高度从根节点轨迹的 Z 坐标中减去，使得所有动作的地面均为 Z=0 平面。

以上计算完初始关节点的后处理包括：

1. 按照法向量计算世界坐标系旋转，对根节点旋转和根节点位移应用此旋转，或直接对计算好的所有关节应用此旋转，矫正动作第一帧的方向；
2. 计算地面高度，修正根节点位移高度，使所有运动地面高度一致；
3. 将 SMPL-X 的轴角姿态表示转化为 6D 旋转表示。

### 2.4 数据集

使用 `dataset/dance_dataset.py` 中的 `Sliced_Motion_Music` 类加载动作和音乐数据。加载数据集时，会对数据进行 [关节点计算](#23-关节点计算) 中描述的预处理和后处理，使得从数据集中获取的数据表示一条标准的运动片段（1. 地面高度为0；2. 运动起始坐标向 XOY 平面投影到原点；3. 运动起始姿态面向 X+ 方向）。经过处理计算得到的数据将会备份至 `backup_data.npz` 文件，可以通过设置 `use_cached = True` 跳过预处理、前向运动学计算和后处理，直接从备份文件中读取处理好的数据。

通过数据集访问数据时，其 `__getitem__()` 方法返回一个字典：

```Python
data_dict = {
  "name":,      # str 序列名称
  "position":,  # (150, 3) tensor 根节点轨迹
  "motion":,    # (150, 24 * 6) tensor 关节点相对旋转 6d repr
  "contact":,   # (150, 4) tensor 脚部接触 4 个相关节点
  "music":,     # (150, 35) tensor 音乐特征
}
```

### 2.5 渲染动作

使用 `visualize/render.py` 中的 `skeleton_render` 函数渲染动作序列，渲染结果将保存为 gif 文件。

```Python
skeleton_render(joints, name=slice_name, out="renders/")
```

</details>

## 3. 模型

TODO List

- [ ] nn.Module 模型
  - [ ] 参数
  - [ ] 输入输出
- [ ] 扩散模型
