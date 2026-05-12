# Problem Formulation: PointPillar-CoAlign-Hybrid-V2XViT-Deform-RSU-Distance

本文档参考 CoAlign 论文的协同 3D 检测问题定义，并结合
`opencood/hypes_yaml/v2xset/lidar_only_with_noise/rago/pointpillar_coalign_hybrid_v2xvit_deform_rsu_distance.yaml`
中的模型设置，形式化描述当前模型的目标、输入、位姿校正、混合特征融合、RSU 距离引导和训练优化问题。

## 1. 协同感知设定

考虑一个 V2X 场景中共有 $N$ 个协同智能体，包括车辆端 CAV 和路侧 RSU。对第 $i$ 个 ego agent，其通信范围由配置设为
$R_{\mathrm{comm}}=100\mathrm{m}$，可通信智能体集合为

$$
\mathcal{N}_i=\left\{j \mid \left\|\mathbf{p}_j-\mathbf{p}_i\right\|_2
\le R_{\mathrm{comm}}\right\}, \qquad |\mathcal{N}_i|\le 5 .
$$

每个智能体 $j\in\mathcal{N}_i$ 观测到 LiDAR 点云
$\mathbf{O}_j$，并由定位模块给出带噪声的位姿

$$
\tilde{\xi}_j = \xi_j \circ \epsilon_j, \qquad
\epsilon_j=(\epsilon_x,\epsilon_y,\epsilon_\theta),
$$

其中配置中的噪声为

$$
\epsilon_x,\epsilon_y \sim \mathcal{N}(0,0.2^2), \qquad
\epsilon_\theta \sim \mathcal{N}(0,0.2^2).
$$

模型目标是：在无需真实位姿监督的情况下，利用单车检测框构建 CoAlign agent-object pose graph 修正相对位姿，再对多个智能体的 PointPillar BEV 特征进行多尺度融合，最终在 ego 坐标系下预测 3D 检测结果

$$
\mathbf{B}'_i=
\left\{
(x,y,z,l,w,h,\theta,s)
\right\}.
$$

## 2. CoAlign 总体问题

遵循 CoAlign 的混合协同范式，每个智能体同时传输中间特征和单体检测结果。对 ego agent $i$，当前模型可写为

$$
\begin{aligned}
\left(\{\mathbf{F}_j^{(\ell)}\}_{\ell=1}^{L}, \mathbf{B}_j\right)
&= f_{\mathrm{PP}}(\mathbf{O}_j), \\
\{\xi'_{j\rightarrow i}\}_{j\in\mathcal{N}_i}
&= f_{\mathrm{align}}\left(
\{(\mathbf{B}_j,\tilde{\xi}_j)\}_{j\in\mathcal{N}_i}
\right), \\
\mathbf{M}_{j\rightarrow i}^{(\ell)}
&= \mathcal{W}\left(
\mathbf{F}_j^{(\ell)}, \xi'_{j\rightarrow i}
\right), \\
\mathbf{Z}_i
&= f_{\mathrm{hybrid}}\left(
\{\mathbf{M}_{j\rightarrow i}^{(\ell)}\}_{j\in\mathcal{N}_i,\ell=1}^{L}
\right), \\
\mathbf{B}'_i
&= f_{\mathrm{head}}(\mathbf{Z}_i).
\end{aligned}
$$

其中 $f_{\mathrm{PP}}$ 是 PointPillar encoder 和 ResNet BEV backbone，$L=3$ 为配置中的三层多尺度 BEV 特征，
$\mathcal{W}(\cdot)$ 表示利用修正后相对位姿的仿射特征变换，$f_{\mathrm{hybrid}}$ 是多尺度 attention fusion、V2XViT-Deform 高层增强以及 RSU 距离先验融合组成的混合融合模块。

## 3. PointPillar 特征编码

LiDAR 点云首先按体素大小

$$
\Delta=(0.4,0.4,4)
$$

在范围

$$
[-140.8,-40,-3,140.8,40,1]
$$

内进行 pillar/voxel 化。PillarVFE 将第 $j$ 个智能体的点云编码为 BEV 特征

$$
\mathbf{X}_j=f_{\mathrm{vfe}}(\mathbf{O}_j),
\qquad
\mathbf{X}_j\in\mathbb{R}^{64\times H\times W}.
$$

随后 ResNet BEV backbone 产生三层多尺度特征

$$
\mathbf{F}_j^{(\ell)}
=g_\ell(\mathbf{X}_j), \qquad
\ell\in\{1,2,3\},
$$

对应配置中的通道数

$$
C_1=64,\qquad C_2=128,\qquad C_3=256.
$$

## 4. Agent-Object Pose Graph 位姿校正

CoAlign 的关键思想是利用多个智能体检测到的同一物体作为 landmark，构造二部图

$$
G=(\mathcal{V}^{\mathrm{agent}},
\mathcal{V}^{\mathrm{object}},
\mathcal{E}),
$$

其中 agent 节点为 $\{\xi_j\}$，object 节点为 $\{\chi_k\}$。若智能体 $j$ 检测到物体 $k$，则建立边
$(j,k)\in\mathcal{E}$，边测量为单体检测框给出的相对位姿

$$
z_{jk}=(\hat{x}_{jk},\hat{y}_{jk},\hat{\theta}_{jk}).
$$

该边的位姿一致性误差定义为

$$
\mathbf{e}_{jk}
=z_{jk}^{-1}\circ
\left(\xi_j^{-1}\circ\chi_k\right)
\in\mathbb{R}^{3},
$$

其中 $\circ$ 表示 $SE(2)$ 位姿复合。若检测框带有不确定性
$(\sigma_x^2,\sigma_y^2,\sigma_\theta^2)$，信息矩阵为

$$
\mathbf{\Omega}_{jk}
=\mathrm{diag}
\left(
\frac{1}{\sigma_x^2},
\frac{1}{\sigma_y^2},
\frac{1}{\sigma_\theta^2}
\right).
$$

位姿图优化问题为

$$
\{\xi'_j,\chi'_k\}
=
\underset{\{\xi_j,\chi_k\}}{\arg\min}
\sum_{(j,k)\in\mathcal{E}}
\mathbf{e}_{jk}^{\top}
\mathbf{\Omega}_{jk}
\mathbf{e}_{jk}.
$$

优化时固定 ego 位姿，并更新其它智能体和物体节点。最终得到从智能体 $j$ 到 ego $i$ 的修正相对位姿

$$
\xi'_{j\rightarrow i}
=
(\xi'_i)^{-1}\circ\xi'_j,
\qquad
\xi'_{i\rightarrow i}=I.
$$

在当前 YAML 中，`box_align` 使用预计算的 stage1 boxes，并启用
`use_uncertainty: true`、`landmark_SE2: true`、`abandon_hard_cases: true` 和
`drop_hard_boxes: true`，因此上述位姿图校正可视为本模型融合前的对齐先验。

## 5. 位姿对齐与多尺度 Attention Fusion

给定修正后的相对位姿 $\xi'_{j\rightarrow i}$，第 $\ell$ 层特征被 warp 到 ego 坐标系：

$$
\mathbf{M}_{j\rightarrow i}^{(\ell)}
=
\mathcal{W}_{\ell}
\left(
\mathbf{F}_j^{(\ell)}, \xi'_{j\rightarrow i}
\right),
\qquad
\mathbf{M}_{i\rightarrow i}^{(\ell)}
=\mathbf{F}_i^{(\ell)}.
$$

对每个 BEV 位置 $\mathbf{u}$，多智能体 attention fusion 沿 agent 维进行缩放点积注意力：

$$
\alpha_{ij}^{(\ell)}(\mathbf{u})
=
\frac{
\exp\left(
\frac{
\mathbf{q}_{i}^{(\ell)}(\mathbf{u})^\top
\mathbf{k}_{j}^{(\ell)}(\mathbf{u})
}{
\sqrt{C_\ell}
}
\right)
}{
\sum_{m\in\mathcal{N}_i}
\exp\left(
\frac{
\mathbf{q}_{i}^{(\ell)}(\mathbf{u})^\top
\mathbf{k}_{m}^{(\ell)}(\mathbf{u})
}{
\sqrt{C_\ell}
}
\right)
},
$$

$$
\mathbf{A}_{i}^{(\ell)}(\mathbf{u})
=
\sum_{j\in\mathcal{N}_i}
\alpha_{ij}^{(\ell)}(\mathbf{u})
\mathbf{v}_{j}^{(\ell)}(\mathbf{u}).
$$

其中 $\mathbf{A}_{i}^{(\ell)}$ 是第 $\ell$ 层 CoAlign multiscale attention 融合后的特征。

## 6. Hybrid V2XViT-Deform 高层增强

配置中 `hybrid_v2xvit_deform.level: 2`，因此只在最高语义层 $\ell^\star=3$ 上加入 V2XViT-Deform 分支。先使用 V2X-ViT transformer 对已经对齐的多智能体特征进行全局/窗口注意力融合：

$$
\mathbf{H}_{i}^{(\ell^\star)}
=
f_{\mathrm{V2XViT}}
\left(
\{\mathbf{M}_{j\rightarrow i}^{(\ell^\star)}\}_{j\in\mathcal{N}_i}
\right).
$$

随后使用单层 deformable attention refiner。对每个 BEV query 位置 $\mathbf{u}$、head $h$ 和采样点 $p$，模型预测偏移
$\Delta\mathbf{u}_{hp}$ 和权重 $a_{hp}$：

$$
\tilde{\mathbf{H}}_{i}^{(\ell^\star)}(\mathbf{u})
=
\sum_{h=1}^{H_a}
\sum_{p=1}^{P}
a_{hp}(\mathbf{u})\,
\mathbf{W}_{h}
\mathbf{H}_{i}^{(\ell^\star)}
\left(
\mathbf{u}+\Delta\mathbf{u}_{hp}(\mathbf{u})
\right),
$$

其中配置为 $H_a=4$、$P=4$。混合分支通过可学习门控与原 attention fusion 结果保守融合：

$$
\gamma=\sigma(\beta), \qquad \beta_{\mathrm{init}}=-4.0,
$$

$$
\bar{\mathbf{A}}_{i}^{(\ell^\star)}
=
\mathbf{A}_{i}^{(\ell^\star)}
+
\gamma
\left(
\tilde{\mathbf{H}}_{i}^{(\ell^\star)}
-
\mathbf{A}_{i}^{(\ell^\star)}
\right).
$$

对非增强尺度，有

$$
\bar{\mathbf{A}}_{i}^{(\ell)}
=
\mathbf{A}_{i}^{(\ell)},
\qquad
\ell\ne \ell^\star.
$$

## 7. RSU 距离引导语义细化

配置中 `rsu_index: -1`，即默认每个场景中的最后一个 agent 为 RSU。RSU 距离引导同样作用于最高语义层
$\ell_r=3$。设 $\mathbf{R}_{i}^{(\ell_r)}$ 为 RSU 特征 warp 到 ego 坐标后的结果：

$$
\mathbf{R}_{i}^{(\ell_r)}
=
\mathcal{W}_{\ell_r}
\left(
\mathbf{F}_{\mathrm{rsu}}^{(\ell_r)},
\xi'_{\mathrm{rsu}\rightarrow i}
\right).
$$

对 BEV 网格位置 $\mathbf{u}=(x,y)$，以配置中的中心
$(c_x,c_y)=(0,0)$ 计算距离

$$
d(\mathbf{u})
=
\sqrt{(x-c_x)^2+(y-c_y)^2}.
$$

距离先验在 $[40\mathrm{m},90\mathrm{m}]$ 内线性增大，并裁剪到 $[0,1]$：

$$
\rho(\mathbf{u})
=
\mathrm{clip}
\left(
\frac{d(\mathbf{u})-40}{90-40},
0,1
\right).
$$

基础 RSU 融合权重为

$$
\alpha_0(\mathbf{u})
=
\alpha_{\min}
+
(\alpha_{\max}-\alpha_{\min})\rho(\mathbf{u}),
\qquad
\alpha_{\min}=0,\quad \alpha_{\max}=0.7.
$$

代码中还引入可学习缩放和偏置：

$$
s=\sigma(\eta),\qquad b=\sigma(\mu),
\qquad
\eta_{\mathrm{init}}=1.5,\quad \mu_{\mathrm{init}}=-2.0,
$$

$$
\alpha_{\mathrm{rsu}}(\mathbf{u})
=
\mathrm{clip}
\left(
s\alpha_0(\mathbf{u})+b,
0,1
\right).
$$

最终将融合后的高层语义特征向 RSU 特征插值：

$$
\hat{\mathbf{A}}_{i}^{(\ell_r)}(\mathbf{u})
=
\bar{\mathbf{A}}_{i}^{(\ell_r)}(\mathbf{u})
+
\alpha_{\mathrm{rsu}}(\mathbf{u})
\left(
\mathbf{R}_{i}^{(\ell_r)}(\mathbf{u})
-
\bar{\mathbf{A}}_{i}^{(\ell_r)}(\mathbf{u})
\right).
$$

直观上，距离越远，车辆侧 LiDAR 越容易稀疏和遮挡，因此模型更倾向于引入 RSU 高视角/远距离语义信息。

## 8. 多尺度解码与检测头

三层融合特征经 backbone 的 deblock 上采样到统一 BEV 尺度后拼接：

$$
\mathbf{Z}_i
=
\mathrm{Shrink}
\left(
\mathrm{Cat}
\left[
u_1(\hat{\mathbf{A}}_{i}^{(1)}),
u_2(\hat{\mathbf{A}}_{i}^{(2)}),
u_3(\hat{\mathbf{A}}_{i}^{(3)})
\right]
\right).
$$

其中 `shrink_header.input_dim=384`，输出通道为 $256$。检测头分别输出分类、回归和方向预测：

$$
\mathbf{P}_i
=
f_{\mathrm{cls}}(\mathbf{Z}_i),
\qquad
\mathbf{T}_i
=
f_{\mathrm{reg}}(\mathbf{Z}_i),
\qquad
\mathbf{D}_i
=
f_{\mathrm{dir}}(\mathbf{Z}_i).
$$

配置中 anchor yaw 为 $0^\circ$ 和 $90^\circ$，每个位置有 $A=2$ 个 anchor，因此

$$
\mathbf{P}_i\in\mathbb{R}^{A\times H\times W},
\qquad
\mathbf{T}_i\in\mathbb{R}^{7A\times H\times W},
\qquad
\mathbf{D}_i\in\mathbb{R}^{2A\times H\times W}.
$$

## 9. 训练目标

当前 YAML 使用 `point_pillar_loss`，由 focal classification loss、Smooth-L1 box regression loss 和 direction classification loss 组成：

$$
\mathcal{L}
=
\lambda_{\mathrm{cls}}\mathcal{L}_{\mathrm{focal}}
+
\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{smoothL1}}
+
\lambda_{\mathrm{dir}}\mathcal{L}_{\mathrm{dir}},
$$

其中配置给出

$$
\lambda_{\mathrm{cls}}=2.0,\qquad
\lambda_{\mathrm{reg}}=2.0,\qquad
\lambda_{\mathrm{dir}}=0.2.
$$

分类项采用 sigmoid focal loss：

$$
\mathcal{L}_{\mathrm{focal}}
=
-
\alpha_t
(1-p_t)^{\gamma}
\log(p_t),
\qquad
\alpha=0.25,\quad \gamma=2.
$$

回归项只在正 anchor 上计算，并对角度采用 sine difference：

$$
\Delta\theta
=
\sin(\hat{\theta})\cos(\theta)
-
\cos(\hat{\theta})\sin(\theta)
=
\sin(\hat{\theta}-\theta).
$$

方向项使用两个方向 bin 的 softmax cross entropy：

$$
\mathcal{L}_{\mathrm{dir}}
=
\mathrm{CE}
\left(
\mathbf{D}_i,\mathbf{D}_i^\star
\right).
$$

因此，本模型最终求解的问题可以概括为

$$
\theta^\star
=
\underset{\theta}{\arg\min}\;
\mathbb{E}_{(\mathbf{O},\mathbf{Y})}
\left[
\mathcal{L}
\left(
f_\theta
\left(
\{\mathbf{O}_j,\tilde{\xi}_j,\mathbf{B}^{\mathrm{stage1}}_j\}_{j\in\mathcal{N}_i}
\right),
\mathbf{Y}_i
\right)
\right],
$$

其中 $f_\theta$ 包含 PointPillar 编码器、多尺度 CoAlign attention fusion、V2XViT-Deform 高层增强、RSU 距离引导融合和检测头；$\mathbf{B}^{\mathrm{stage1}}_j$ 为配置中 `box_align` 指定的预计算单体检测框，用于提供位姿图对齐约束。

## 10. 与原 CoAlign 的对应关系

原 CoAlign 的 problem formulation 关注：带噪声位姿下，多智能体通过中间特征和单体检测框协同，利用 agent-object pose graph 校正相对位姿，再进行多尺度中间融合。当前 YAML 对应模型在此基础上加入两点扩展：

1. 在最高语义尺度引入 V2XViT-Deform 分支，用 transformer 和 deformable attention 对远程协同语义进行增强，并通过小初值门控 $\gamma=\sigma(-4)$ 保持训练初期稳定。
2. 在最高语义尺度引入 RSU 距离先验 $\alpha_{\mathrm{rsu}}(\mathbf{u})$，使远距离区域更偏向 RSU 对齐后的语义特征。

因此，该模型的核心问题不是单纯的 3D 检测，而是：

$$
\textbf{在存在定位噪声的 V2X 场景中，联合利用 CoAlign 位姿一致性、车辆/RSU 多尺度特征互补、V2XViT-Deform 语义增强和距离先验，学习一个对远距离与位姿误差更鲁棒的 ego 端 3D 检测函数。}
$$

