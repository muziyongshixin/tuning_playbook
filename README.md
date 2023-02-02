# Deep Learning Tuning Playbook

*This is not an officially supported Google product.*

**Varun Godbole<sup>&dagger;</sup>, George E. Dahl<sup>&dagger;</sup>, Justin Gilmer<sup>&dagger;</sup>, Christopher J. Shallue<sup>&Dagger;</sup>, Zachary Nado<sup>&dagger;</sup>**


&dagger; Google Research, Brain Team

&Dagger; Harvard University

## Table of Contents

-   [这个文档适合谁？](#这个文档适合谁？)
-   [Why a tuning playbook?](#why-a-tuning-playbook)
-   [Guide for starting a new project](#guide-for-starting-a-new-project)
    -   [Choosing the model architecture](#choosing-a-model-architecture)
    -   [Choosing the optimizer](#choosing-the-optimizer)
    -   [Choosing the batch size](#choosing-the-batch-size)
    -   [Choosing the initial configuration](#choosing-the-initial-configuration)
-   [A scientific approach to improving model performance](#a-scientific-approach-to-improving-model-performance)
    -   [The incremental tuning strategy](#the-incremental-tuning-strategy)
    -   [Exploration vs exploitation](#exploration-vs-exploitation)
    -   [Choosing the goal for the next round of experiments](#choosing-the-goal-for-the-next-round-of-experiments)
    -   [Designing the next round of experiments](#Designing-the-next-round-of-experiments)
    -   [Determining whether to adopt a training pipeline change or
        hyperparameter
        configuration](#Determining-whether-to-adopt-a-training-pipeline-change-or-hyperparameter-configuration)
    -   [After exploration concludes](#After-exploration-concludes)
-   [Determining the number of steps for each training run](#Determining-the-number-of-steps-for-each-training-run)
    -   [Deciding how long to train when training is not compute-bound](#Deciding-how-long-to-train-when-training-is-not-compute-bound)
    -   [Deciding how long to train when training is compute-bound](#Deciding-how-long-to-train-when-training-is-compute-bound)
-   [Additional guidance for the training pipeline](#Additional-guidance-for-the-training-pipeline)
    -   [Optimizing the input pipeline](#Optimizing-the-input-pipeline)
    -   [Evaluating model performance](Evaluating-model-performance)
    -   [Saving checkpoints and retrospectively selecting the best checkpoint](#Saving-checkpoints-and-retrospectively-selecting-the-best-checkpoint)
    -   [Setting up experiment tracking](#Setting-up-experiment-tracking)
    -   [Batch normalization implementation details](#Batch-normalization-implementation-details)
    -   [Considerations for multi-host pipelines](#Considerations-for-multi-host-pipelines)
-   [FAQs](#faqs)
-   [Acknowledgments](#acknowledgments)
-   [Citing](#citing)
-   [Contributing](#contributing)

## 这个文档适合谁？

本文档适用于有兴趣**最大化深度学习模型性能**的工程师和研究人员(包括个人和团队)。我们假设有机器学习和深度学习概念的基本知识。

我们的重点是**超参数调优的过程**。我们谈到了深度学习训练的其他方面，比如管道实现和优化，但我们对这些方面的处理并不打算是完整的。

我们假设机器学习问题是一个监督学习问题，或者看起来很像监督学习问题(例如，自监督)。也就是说，本文件中的一些处方也可能适用于其他类型的问题。
 

## 为什么要写这样一个文档？

目前，要让深度神经网络在实践中很好地工作，需要进行大量的辛劳和猜测。更糟糕的是，人们使用深度学习来获得良好结果的实际方法很少被记录下来。论文掩盖了导致最终结果的过程，以呈现一个更清晰的故事，而研究商业问题的机器学习工程师很少有时间退一步，概括他们的过程。教科书倾向于回避实际指导，优先考虑基本原则，即使它们的作者在应用工作中有必要的经验，可以提供有用的建议。在准备创建本文档时，我们找不到任何全面的尝试来真正解释如何使用深度学习获得良好的结果。相反，我们在博客文章和社交媒体上找到了一些建议的片段，在研究论文的附录中发现了一些技巧，偶尔会有关于某个特定项目或管道的案例研究，还有很多困惑。深度学习专家和不太熟练的从业者使用表面上相似的方法所取得的结果之间存在着巨大的鸿沟。与此同时，这些专家欣然承认，他们所做的一些事情可能并不完全合理。随着深度学习的成熟并对世界产生更大的影响，社区需要更多的资源来涵盖有用的菜谱，包括所有对获得良好结果至关重要的实际细节。

我们是一个由五名研究人员和工程师组成的团队，他们在深度学习领域工作了多年，其中一些人早在2006年就开始了。我们已经将深度学习应用于从语音识别到天文学的所有问题，并在此过程中学到了很多东西。这份文档源于我们自己训练神经网络的经验，教授新的机器学习工程师，并就深度学习的实践为我们的同事提供建议。尽管看到深度学习从少数学术实验室实践的机器学习方法发展为数十亿人使用的产品技术是令人欣慰的，但深度学习作为一门工程学科仍处于起步阶段，我们希望这份文件鼓励其他人帮助系统化该领域的实验协议。

这份文件的出现是为了明确我们自己的深度学习方法，因此它代表了作者在写作时的观点，而不是任何形式的客观事实。我们自己在超参数调优方面的挣扎使它成为我们指南的一个特别重点，但我们也涵盖了我们在工作中遇到的其他重要问题(或看到的错误)。我们的意图是让这项工作成为一份活的文件，随着我们信念的改变而成长和发展。例如，关于调试和减轻训练失败的材料在两年前是不可能写出来的，因为它是基于最近的结果和正在进行的调查。不可避免地，我们的一些建议将需要更新，以说明新的结果和改进的工作流程。我们不知道最佳的深度学习配方，但在社区开始写下并讨论不同的过程之前，我们无法指望找到它。为此，我们鼓励那些对我们的建议有异议的读者提出替代建议，并提供令人信服的证据，这样我们就可以更新剧本。我们也很乐意看到可能有不同建议的替代指南和剧本，这样我们就可以作为一个社区努力实现最佳实践。最后，任何标有🤖表情符号的区域都是我们想做更多研究的地方。只有在尝试写完这本剧本之后，我才完全清楚在深度学习从业者的工作流程中可以找到多少有趣而被忽视的研究问题。

## 如何开始一个新的project？

我们在调优过程中所做的许多决定可以在项目开始时一次性做出，只有在环境发生变化时才会偶尔重新进行调整。

我们假设开始处理该问题时应该满足以下条件：
-   问题定义已经清晰，已经完成了足够的数据清理等基本工作，因此在模型架构和训练配置上花费时间是有意义的。
-   已经建立了一个进行训练和评估的流程，并且很容易为各种感兴趣的模型执行训练和预测工作。
-   已经选择并实现了适当的评价指标。这些指标应该在部署环境中能够评估。

### 选择模型架构

***总结:***  *通常开始一个新项目的时候，优先选择一个已经确定能够work的模型*

- 优先选择一个被广泛使用的模型，并且让他能够跑起来。因为后面去做自己的定制化模型通常来说难度不大。
- 模型结构通常会因为多种超参数的影响而不同（例如，模型层数，宽度，激活函数的种类）
    - 因此，选择架构实际上意味着选择一系列不同的模型(每个模型超参数设置对应一个模型)。
    - 我们将在后面讨论如何设置模型部分的超参数 [Choosing the initial configuration](#choosing-the-initial-configuration)
        和
        [A scientific approach to improving model performance](#a-scientific-approach-to-improving-model-performance).
- 如果可能的话，找一篇和你要处理的问题足够相似的paper，并且复现他，并以该模型作为起点去做后续的优化。

### 选择合适的Optimizer

***Summary:***  *从针对当前问题的最流行的优化器开始*

   
- 目前来说并没有一个Optimizer对于所有的机器学习问题是最好的，因为就算单纯的[比较Optimizer的性能都是一个困难的问题](https://arxiv.org/abs/1910.05446). 🤖
- 我们推荐在开始一个新的项目的时候优先选择广泛使用的的Optimizer。
    - 理想情况下选择一个该领域下的最流行的Optimizer。
=
- 需要注意准备好关注所选优化器的**\*****所有****\***超参数。

    - 拥有更多超参数的Optimizer往往需要更多的时间来找到最优配置。
    - 这在项目的开始阶段特别相关，在这个阶段，我们试图找到各种超参数（例如架构超参数）的最佳值，同时将优化器超参数视为[干扰参数](#identifying-scientific-nuisance-and-fixed-hyperparameters).。
    - 在项目开始的时候，优先选择一个简单一点的优化器（例如固定momentum参数的SGD，或者固定 $\epsilon$, $\beta_{1}$,  $\beta_{2}$ 等参数的Adam优化器），之后再切换到稍微复杂一些的优化器。
- 
-   我们喜欢的优秀的优化器包括（但不限于）:
    -   [SGD with momentum](#what-are-the-update-rules-for-all-the-popular-optimization-algorithms)
        (我们喜欢Nesterov变体)
    -   [Adam and NAdam](#what-are-the-update-rules-for-all-the-popular-optimization-algorithms),
        它们比具有动量的SGD更通用。请注意，Adam有4个可调超参数，
        [他们每一个都很重要](https://arxiv.org/abs/1910.05446)!
        -   查看[How should Adam's hyperparameters be tuned?](#how-should-adams-hyperparameters-be-tuned)

### 选择Batchsize的大小

***Summary:***  * Batchsize大小决定了训练速度，但不应该用来直接调优验证集的性能。理想的批大小通常是可用硬件支持的最大Batchsize。*

-   Batchsize的大小通常是决定*训练时间* 和*训练消耗资源*的关键参数。
- 增加Batchsize的大小通常可以减少训练时间，这通常是有益的，因为：
    - 可以在固定的时间内尽可能多的调节超参数，从而得到一个更好的模型。
    - 可以降低部署的延迟，可以让新的idea更快得到验证。
   
- 增加Batchsize的大小可能增加也可能减少计算消耗，但是也可能没有任何影响。
- 不应该将Batchsize当做一个可优化的超参数来调整验证集上的性能。
    - 当所有的超参数被设置好（特别是lr和正则化超参数），并且训练足够的时间，模型最终的性能跟Batchsize关系不大。(参考[Shallue et al. 2018](https://arxiv.org/abs/1811.03600)).
    -   其他查看 [Why shouldn't the batch size be tuned to directly improve
        validation set
        performance?](#why-shouldnt-the-batch-size-be-tuned-to-directly-improve-validation-set-performance)

#### 确定可行的Batchsize大小和估计训练吞吐量

<details><summary><em>[Click to expand]</em></summary>

<br>

- 对于给定的模型和优化器，可支持的Batchsize通常有一定的范围，受到可用硬件的限制。通常，限制因素是GPU或加速器内存。
- 不幸的是，在没有运行（或至少编译）完整的训练程序的情况下，计算哪些Batchsize可以适合内存可能很困难。
最简单的解决方案通常是在不同的Batchsize（例如以2的幂数增加）运行训练作业，直到其中一个作业超出可用内存为止。
- 对于每个Batchsize，我们应该训练足够长的时间，以获得可靠的训练吞吐量（*training throughput*）估计。


<p align="center">training throughput = (# examples processed per second)</p>

<p align="center">or, equivalently, the <em>time per step</em>.</p>

<p align="center">time per step = (batch size) / (training throughput)</p>

- 当加速器尚未饱和时，如果Batchsize大小翻倍，训练吞吐量也应该翻倍（或者至少接近翻倍）。 等价地，当Batchsize增加时，每步骤的时间应该是恒定的（或者至少接近恒定）。
- 如果不是这种情况，那么训练管道就有一个瓶颈，例如计算节点之间的I / O或同步。 在继续之前，我们需要找出问题所在。
- 如果训练吞吐量仅增加到某个最大Batchsize，则我们应该仅考虑这个最大Batchsize以内的Batchsize，即使硬件支持更大的Batchsize。（例如Batchsize从128 调整到256吞吐量没有增加，那么即使GPU支持更大的Batchsize，我们也应该使用128作为合适的Batchsize。）
- 使用更大Batchsize的所有优势都假设训练吞吐量增加。如果没有，请修复瓶颈或使用较小的Batchsize。
- 梯度累积模拟硬件无法支持的更大Batchsize，因此不提供任何吞吐量效益。在实际工作中通常应该避免使用。
- 每次更改模型或优化器时（例如，不同的模型架构可以允许更大的Batchsize），都可能需要重复这些步骤。

</details>

#### 选择Batchsize以最大程度地减少训练时间

<details><summary><em>[Click to expand]</em></summary>

<br>


<p align="center">Training time = (time per step) x (total number of steps)</p>

-  通常，我们可以认为对于所有可行的Batchsize，每步时间大致相同。当没有并行计算的开销，所有训练瓶颈都已诊断和修正时，这是正确的（有关如何识别训练瓶颈，请参见[上一节](#确定可行的Batchsize大小和估计训练吞吐量)）。
实际上，随着Batchsize的增加，通常至少会有一些开销。
- 随着Batchsize的增加，需要达到固定性能目标的总步数通常会减少（前提是当Batchsize更改时所有相关超参数都已重新调整；参见[Shallue et al. 2018](https://arxiv.org/abs/1811.03600)）。
    - 例如，将Batchsize翻倍可能会将所需总步数减半。这称为**完美缩放**。
    - 对于所有Batchsize，完美缩放适用于关键Batchsize之前，超出该关键Batchsize大小后收益逐渐减少。
    - 最终，增加Batchsize大小不再减少训练步数（但也不会增加）。
- 因此，最小化训练时间的Batchsize通常是可以减少训练步骤数量的最大Batchsize。

    - 这个Batchsize取决于数据集，模型和优化器，如何计算它，而不是为每个新问题进行实验性查找，是一个悬而未决的问题。 🤖
    - 在比较Batchsize时，请注意[样本预算/时期预算](https://developers.google.com/machine-learning/glossary#epoch)（在保持训练样本展示数量不变的情况下运行所有实验）和步骤预算（在保持训练步骤数量不变的情况下运行所有实验）之间的区别。
    - 使用样本预算比较Batchsize只会探测完美缩放，即使更大的Batchsize仍然可以通过减少训练步骤数量来提供有意义的加速。
    - 通常，可用硬件支持的最大Batchsize将小于关键Batchsize。因此，一个很好的经验法则（在不进行任何实验的情况下）是使用尽可能大的Batchsize。
- 如果使用更大的Batchsize最终增加了训练时间，那么使用更大的Batchsize是没有意义的。


</details>

#### 选择Batchsize以最小化资源使用量

<details><summary><em>[Click to expand]</em></summary>

<br>

- 增加 batch size 的时候，与其相关的资源成本有两种：
    1. 前期成本，例如购买新硬件或者重写训练流程以实现多 GPU/多 TPU 训练。
    2. 使用成本，例如团队资源预算、云提供商计费、电力/维护成本。
- 如果增加 batch size 的前期成本很高，那么最好等项目成熟后再评估成本效益权衡。实现多主机并行训练程序可能会带来 bugs 和 其他诡异的问题，所以最好先使用简单的流程。(另一方面，当需要大量的调试实验时，训练时间的大幅加快可能非常有益)。
- 我们称总的使用成本（可能包括多种不同类型的成本）为“资源消耗”。我们可以将资源消耗分解为以下组成部分：

<p align="center">Resource consumption = (resource consumption per step) x (total number of steps)</p>

- 增加Batchsize通常允许我们减少总步数。资源消耗的增加或减少取决于每步的消耗如何变化。

- 增加Batchsize可能会降低资源消耗。例如，如果更大批次的每一步都可以在与较小Batchsize相同的硬件上运行（每步只增加很少的时间），那么每步的资源消耗增加可能被步数减少的降低所抵消。
- 增加Batchsize可能不会改变资源消耗。例如，如果将Batchsize加倍会减少一半的步数并使用两倍的GPU，则总消耗（以GPU小时计）不会改变。
- 增加Batchsize可能会增加资源消耗。例如，如果增加Batchsize需要升级硬件，每步的消耗增加可能会超过步数的减少。


</details>

####  通常更改Batchsize需要重新搜索绝大部分超参数。

<details><summary><em>[Click to expand]</em></summary>

<br>

- 大多数超参数的最佳值对Batchsize敏感。因此，改变Batchsize通常需要重新开始调参过程。
- 优化器超参数与Batchsize最密切相关，因此必须针对每个Batchsize分别调整(例如学习率，动量momentum）和正则化超参数。
- 在开始项目时选择Batchsize时要注意这一点。如果需要稍后更改Batchsize，重新为新的Batchsize调整所有内容可能困难，耗时和昂贵。


</details>

#### BatchNorm(BN) 跟Batchsize的关系

<details><summary><em>[Click to expand]</em></summary>

<br>


-   批量范数（BN）比较复杂，一般来说，应该使用不同于梯度计算的Batchsize来计算统计信息。有关详细讨论，请参阅[批处理规范部分](#batch-normalization-implementation-details)。

</details>

### 选择初始配置


- 在开始超参数调优之前，我们必须确定起始点。这包括指定(1)模型配置(例如层数)，(2)优化器超参数(例如学习率)，以及(3)训练迭代次数。

- 确定这个初始配置将需要一些手动配置的训练运行和试错。

- 我们的指导原则是找到一个简单、相对快速、相对低资源消耗的配置，从而获得“合理”的结果。

    - “简单”意味着尽可能避免花哨的东西;这些都可以在以后添加。即使事后证明这些功能是有用的，在初始配置中添加它们也有浪费时间调整无用功能和/或陷入不必要的复杂性的风险。
    - 例如，在添加花哨的衰减（lr decay schedules）之前，先从恒定的学习率开始。
    - 选择一个快速且消耗最少资源的初始配置将使超参数调优更加有效。例如，从一个较小的模型开始。
    - “合理的”性能取决于问题本身，但至少意味着经过训练的模型在验证集上的表现比随机机会要好得多(尽管它可能坏到不值得部署)。

- 选择训练步数涉及到平衡以下问题:

    - 一方面，训练更多的步骤可以提高性能，并使超参数调优更容易(参见[Shallue et al. 2018](https://arxiv.org/abs/1811.03600))。
- 另一方面，训练步骤更少意味着每次训练运行更快，使用更少的资源，通过减少周期之间的时间和允许更多的实验并行运行来提高调优效率。此外，如果一开始选择了一个不必要的大步骤预算，那么在接下来的过程中可能很难改变它，例如，一旦学习率计划针对该步骤数进行了调整。


## 提高模型性能的科学方法

就本文而言，机器学习开发的最终目标是最大化已部署模型的效用。尽管不同应用程序的开发过程在许多方面有所不同(例如，时间长度、可用的计算资源、模型类型)，但我们通常可以在任何问题上使用相同的基本步骤和原则。

我们的指导原则基于以下假设:

- 已经有一个完全运行的训练流程，以及一个获得合理结果的配置。
- 有足够的计算资源来进行有意义的调优实验，并并行运行至少几个训练作业。
 

### The incremental tuning strategy
### 增量调优策略

***总结:*** 从一个简单的配置开始，逐步改进，同时深入了解问题。确保任何改进都是基于强有力的证据，以避免增加不必要的复杂性。

- 我们的最终目标是找到一种配置，使我们的模型的性能最大化。
    - 在某些情况下，我们的目标将是在一个固定的截止日期前最大化我们可以改进模型的程度(例如提交给一个比赛)。
    - 在其他情况下，我们希望无限地改进模型(例如，不断地改进生产中使用的模型)。
- 原则上，我们可以通过使用算法自动搜索可能配置的整个空间来最大化性能，但这不是一个实际可操作的选择。
    - 因为可能配置的空间非常大，目前还没有任何复杂的算法能够在没有人类引导的情况下有效地搜索这整个空间。
- 大多数自动搜索算法依赖于手动设计的搜索空间，该空间定义了要搜索的配置集，而这些搜索空间可能相当重要。
- 最大化性能的最有效方法是从简单的配置开始，逐步添加功能，并在深入了解问题的同时进行改进。
    - 我们在每一轮调优中使用自动搜索算法，并随着我们理解的增长不断更新我们的搜索空间。
- 随着探索，我们自然会发现更好的配置，因此我们的“最佳”模型将不断改进。
    - 当我们更新我们的最佳配置时，我们称之为启动(这可能对应于生产模型的实际启动，也可能不对应)。
    - 对于每次启动，我们必须确保改变是基于强有力的证据——而不是基于幸运配置的随机机会——这样我们就不会给训练流程增加不必要的复杂性。
 
- 概括来说，我们的增量调优策略包括重复以下四个步骤:

1. 为下一轮实验确定一个适当范围的目标。
2. 设计并运行一组朝着这个目标前进的实验。
3. 从结果中学习我们能学到的东西。
4. 考虑是否启动新的最佳配置。

本节的其余部分将更详细地考虑这一策略。

### 探索vs开发
***总结:*** 大多数时候，我们的主要目标是深入了解问题。

- 尽管有人可能会认为我们将花费大部分时间在验证集上试图最大化性能，但实际上我们将花费大部分时间试图深入了解问题，而相对较少的时间贪婪地集中在验证错误上。
    - 换句话说，我们把大部分时间花在了“探索”上，而只花了一小部分时间在“开发”上。
- 从长远来看，如果我们想要最大化我们的最终表现，理解问题是至关重要的。将洞察力置于短期收益之上可以帮助我们:
    - 避免仅仅由于历史事故而在运行良好的运行中出现不必要的更改。
    - 确定验证错误对哪些超参数最敏感，哪些超参数相互作用最多，因此需要一起重新调优，以及哪些超参数对其他变化相对不敏感，因此可以在未来的实验中修复。
    - 建议可以尝试的潜在新特性，例如如果过度拟合是一个问题，可以使用新的正则化器。
    - 识别出没有帮助的特征，因此可以删除，从而降低未来实验的复杂性。
    - 认识到何时超参数调优的改进可能已经饱和。
    - 缩小最优值周围的搜索空间，以提高调优效率。
- 当我们最终准备好贪心策略提升模型性能时，我们可以纯粹地关注验证错误，即使实验不能最大限度地提供关于调优问题结构的信息。

### 选择下一轮实验的目标
***总结:*** 每一轮实验都应该有一个明确的目标，并且在实验范围上要足够窄，使实验能够朝着目标前进。

- 每一轮实验都应该有一个明确的目标，并且范围要足够窄，这样实验才能朝着目标前进:如果我们试图同时添加多个特征或回答多个问题，我们可能无法理清对结果的单独影响。
- 目标包括:
    - 尝试对训练流程进行潜在的改进(例如，一个新的正则化器，预处理选择等)。
    - 理解特定模型超参数的影响(例如激活函数)
    - 尽可能地最大化验证错误。


### 设计下一轮实验

***总结:*** 对于实验目标来说，首先需要确定哪些超参数是科学的超参数（scientific parameters）、哪些是干扰的超参数（nuisance hyperparameters）以及哪些是固定的超参数（fixed parameters）。
在优化干扰超参数的时候来比较不同的科学超参数对实验结果的影响。
同时需要选择干扰参数的搜索空间来平衡资源成本与科学价值。

#### 识别科学的、干扰的和固定的超参数 （scientific, nuisance, and fixed hyperparameters）

<details><summary><em>[Click to expand]</em></summary>

<br>

- 对于一个给定的目标，所有超参数要么是**科学超参数**，要么是**干扰超参数**，要么是**固定超参数**。
    - 科学超参数是那些我们试图测量的对模型性能的影响。
    - 干扰超参是那些为了公平地比较科学超参数的不同值而需要优化的超参数。这类似于妨害参数的统计概念[nuisance parameters](https://en.wikipedia.org/wiki/Nuisance_parameter).。
    - 固定的超参数在本轮实验中会有固定的值。在比较科学超参数的不同值时，这些超参数的值不需要(或者我们不希望)改变。
        - 通过固定一组实验的某些超参数，我们必须接受从实验中得出的结论可能对其他固定超参数的设置无效。换句话说，固定超参数会对我们从实验中得出的任何结论的真实性造成挑战（因为这些参数是固定的，所以一些结论并不一定100%可靠）。
- 例如，如果我们的目标是“确定具有更多隐藏层的模型是否会减少验证错误”，那么隐藏层的数量就是一个科学的超参数。
    - 学习率是一个干扰超参，因为只有在每个层数分别调整学习率的情况下，我们才能公平地比较具有不同隐藏层数的模型(最佳学习率通常取决于模型架构)。
    - 如果我们在之前的实验中确定了激活函数的最佳选择对模型深度不敏感，或者如果我们愿意限制我们关于隐藏层数的结论，只覆盖这个特定的激活函数选择，那么激活函数可以是一个固定的超参数。或者，如果我们准备为每个隐藏层的数量分别调优它，它可能是一个干扰的参数。
- 一个特定的超参数是科学超参数、干扰超参还是固定超参数不是超参数固有的，而是根据实验目标而变化的。
    - 例如，激活函数的选择可以是一个科学的超参数(对于我们的问题，ReLU或tanh是更好的选择吗?)，一个干扰超参(当我们允许几个不同的可能的激活函数时，最好的5层模型比最好的6层模型更好吗?)，或者一个固定的超参数(对于ReLU网络，在特定位置添加批量归一化是否有帮助?)

- 在设计新一轮实验时，我们首先确定实验目标的科学超参数。
    - 在这个阶段，我们认为所有其他超参数都是干扰超参。

- 接下来，我们将一些干扰超参转换为固定的超参数。
    - 如果有了无限的资源，我们会把所有非科学的超参数都当作干扰超参，这样我们从实验中得出的结论就不会受到固定超参数值的限制。
    - 然而，我们试图调优的超参数越多，我们在每次科学超参数设置中调优不够好并最终从实验中得出错误结论的风险就越大。
        - 我们可以通过增加计算预算来应对这种风险，但通常我们的最大资源预算小于调优所有非科学超参数所需的资源预算。
    - 我们选择将一个干扰超参转换为一个固定的超参数，根据我们的判断，固定它所带来的警告比将它作为一个干扰超参所带来的代价要小。
        - 给定的干扰超参与科学超参数的交互作用越多，它对固定其值的破坏就越大。例如，权值衰减强度的最佳值通常取决于模型大小，因此比较不同的模型大小假设一个单一的权值衰减将不是很有见地。

- 尽管我们分配给每个超参数的类型取决于实验目标，但对于某些类别的超参数，我们有以下经验法则:

    - 在各种优化器超参数(例如学习率、动量、学习率计划参数、Adam beta等)中，至少有一些超参数是令人讨厌的干扰参数，因为它们往往与其他变化交互最多。
        - 它们很少是科学的超参数，因为像“当前训练流程的最佳学习率是多少?”这样的目标并不能提供太多的洞见——无论如何，最佳设置很容易随着下一个训练流程更改而更改。
        - 尽管由于资源限制，或者当我们有特别有力的证据表明它们不与科学参数相互作用时，我们可能偶尔会固定其中的一些，但我们通常应该假设优化器超参数必须单独调优，以便在科学超参数的不同设置之间进行公平的比较，因此不应该固定他们。
            - 此外，我们没有先验的正当理由选择一个优化器超参数值而不是另一个(因为它们通常不会影响正向传递或梯度的计算成本)。

    - 相比之下，优化器的选择通常是科学超参数或固定超参数。

        - 如果我们的实验目标是在两个或多个不同的优化器之间进行公平的比较，那么它就是一个科学超参数。“确定哪个优化器在给定数量的步骤中产生的验证错误最低”)。
        - 或者，我们可能出于多种原因使其成为一个固定的超参数，包括:(1)先前的实验使我们相信我们问题的最佳优化器对当前科学的超参数不敏感;或者(2)我们更喜欢使用这个优化器来比较科学超参数的值，因为它的训练曲线更容易推理;(3)我们更喜欢使用这个优化器，因为它比替代方案使用更少的内存。
    - 由正则化技术引入的超参数通常是干扰超参数，但是否使用正则化技术是一个科学的或固定的超参数。

        - 例如，dropout增加了代码复杂度，所以在决定是否包含它时，我们将“no dropout”和“dropout”作为一个科学超参数，而dropout率则是一个干扰超参。
        - 如果我们决定在这个实验的基础上添加dropout到我们的管道中，那么dropout率将是未来实验中一个干扰超参。
    
    - 架构超参数通常是科学的或固定的超参数，因为架构更改会影响服务和训练成本、延迟和内存需求。
        - 例如，层数通常是一个科学的或固定的超参数，因为它往往对训练速度和内存使用有显著的影响。

- 在某些情况下，干扰超参数集和固定超参数集将取决于科学超参数的值。

    - 例如，假设我们试图从Nesterov动量和Adam中确定哪个优化器的验证误差最小。科学超参数是优化器，它接受值`{"Nesterov_momentum", "Adam"}`。`optimizer="Nesterov_momentum"` 引入干扰/固定的超参数`{learning_rate, momentum}`，`optimizer="Adam"`引入干扰/固定的超参数 `{learning_rate, beta1, beta2, epsilon}`.
    - 只对科学超参数的某些值存在的超参数称为条件超参数。
    - 我们不应该仅仅因为两个条件超形参有相同的名称就假定它们是相同的!在上面的例子中，条件超参数`learning_rate`对于optimizer="Nesterov_momentum"和optimizer="Adam"是不同的超参数。它在两种算法中的作用相似(尽管不完全相同)，但是在每个优化器中工作良好的值范围通常有几个数量级的差异。

</details>

#### 创建一系列的实验



<details><summary><em>[Click to expand]</em></summary>

<br>

- 一旦我们确定了科学的和干扰超参，我们就设计了一个“研究”或一系列研究，以朝着实验目标前进。
    - 一个研究指定了一组超参数配置，以用于后续分析。每个配置被称为“试验”（trial）。
    - 创建一项研究通常包括选择在不同试验中不同的超参数，选择这些超参数可以取的值(“搜索空间”)，选择试验的数量，以及选择一个自动搜索算法来从搜索空间中采样这些试验。或者，我们可以通过手动指定超参数配置集来创建一个研究。

- 研究的目的是使用不同的科学超参数值运行管道，同时“优化掉”(或“优化掉”)干扰超参，以便在不同的科学超参数值之间进行尽可能公平的比较。

- 在最简单的情况下，我们将为科学参数的每个配置进行单独的研究，其中每个研究都调整了干扰超参。

    - 例如，如果我们的目标是从Nesterov动量和Adam中选择最好的优化器，我们可以创建一个研究，其中`optimizer="Nesterov_momentum"`，干扰超参为{learning_rate, momentum}，另一个研究，其中`optimizer="Adam"`，干扰超参为{learning_rate, beta1, beta2, epsilon}。我们将通过从每个研究中选择性能最好的试验来比较两种优化器。
    - 我们可以使用任何无梯度优化算法，包括贝叶斯优化或进化算法等方法，来优化干扰超参，尽管我们更喜欢在调优的探索阶段使用准随机搜索，因为它在这种设置中具有各种优势。探索结束后，如果有最先进的贝叶斯优化软件，这是我们的首选。

- 在更复杂的情况下，我们想要比较大量的科学超参数的值，并且进行许多独立的研究是不切实际的，我们可以将科学参数与干扰超参包含在同一个搜索空间中，并使用搜索算法在单个研究中对科学超参数和干扰超参的值进行采样。
    - 采用这种方法时，条件超参数可能会导致问题，因为很难指定搜索空间，除非干扰超参集对于科学超参数的所有值都是相同的。
    - 在这种情况下，我们更倾向于使用准随机搜索而不是花哨的黑盒优化工具，因为它确保我们获得科学超参数的相对均匀的采样值。不管搜索算法是什么，我们都需要确保它能以某种方式统一地搜索科学参数。



</details>

#### 在获取信息和实验成本之间取得平衡



<details><summary><em>[Click to expand]</em></summary>

<br>


- 在设计一项研究或一系列研究时，我们需要分配有限的预算，以充分实现以下三个目标:
    1. 比较足够多的不同科学超参数值。
    2. 在足够大的搜索空间内调优干扰超参。
    3. 对干扰超参的搜索空间进行足够密集的采样。

- 我们越能更好地实现这三个目标，我们就越能从实验中获得更多的见解。
    - 比较尽可能多的科学超参数值拓宽了我们从实验中获得的见解的范围。
    - 包括尽可能多的干扰超参，并允许每个干扰超参在尽可能宽的范围内变化，这增加了我们的信心，即在科学超参数的每个配置的搜索空间中存在一个“好”的干扰超参值。
        - 否则，我们可能会在科学超参数的值之间进行不公平的比较，因为不搜索干扰超参空间的可能区域，其中一些科学参数的值可能存在更好的值。
    - 尽可能密集地对干扰超参的搜索空间进行采样，这样我们才会更有信心找到搜索空间内的最优值。
        - 否则，我们可能会在科学参数的值之间进行不公平的比较，因为一些值随着干扰超参的采样而变得更幸运。
- 不幸的是，这三个维度中的任何一个维度的改进都需要增加试验次数，从而增加资源成本，或者找到一种方法来节省其他维度中的资源。
    - 每个问题都有自己的特点和计算限制，因此如何在这三个需求中分配资源需要一定程度的领域知识。
    - 在进行一项研究后，我们总是试图了解这项研究是否足够好地调整了干扰超参(即搜索了足够大的空间)，以便公平地比较科学的超参数(如下文的更详细描述)。

</details>

### 从实验结果中获取insight

***总结:*** *除了努力实现每组实验的原始科学目标外，还要检查附加问题的清单，如果发现问题，修改实验并重新运行*
 
- 最终，每组实验都有一个特定的目标，我们想要评估实验提供的朝着这个目标的证据。

    - 然而，如果我们提出正确的问题，我们经常会发现需要纠正的问题，然后一组给定的实验才能朝着最初的目标取得很大进展。
        - 如果我们不问这些问题，就可能得出错误的结论。
    - 由于运行实验可能是昂贵的，我们也想借此机会从每组实验中提取其他有用的见解，即使这些见解与当前的目标没有立即相关。

- 在分析一组给定的实验以朝着最初的目标前进之前，我们应该问自己以下额外的问题:
    - 搜索空间是否足够大?
        - 如果一项研究的最佳点在一个或多个维度的搜索空间边界附近，那么搜索可能不够宽。在这种情况下，我们应该进行另一项研究，扩大搜索空间。
    - 我们从搜索空间中采样了足够多的点吗?
        - 如果不是，运行更多的点，或者在调优目标中降低目标。
    - 在每项研究中，有多少比例的试验是不可行的(即试验偏离，得到非常糟糕的损失值，或因为违反了一些隐含的约束而根本无法运行)?
        - 当研究中有很大一部分点是不可行的，我们应该尝试调整搜索空间以避免采样这些点，这有时需要重新参数化搜索空间。
        - 在某些情况下，大量的不可行的点可能表明训练代码中的错误。
    - 模型是否存在优化问题?
    - 我们能从最佳试验的训练曲线中学到什么?
        - 例如，最佳试验的训练曲线是否与有问题的过拟合一致?

- 如有必要，根据上述问题的答案，改进最近的研究(或研究组)，以改善搜索空间和/或抽样更多试验，或采取其他纠正措施。

- 一旦我们回答了上述问题，我们就可以继续评估实验提供的证据，以实现我们最初的目标(例如，评估一个改变是否有用)。


#### 识别错误的搜索空间边界



<details><summary><em>[Click to expand]</em></summary>

<br>

- 如果最佳参数点靠近其边界，则搜索空间是可疑的。如果我们朝那个方向扩展搜索范围，我们可能会发现更优参数值。

- 为了检查搜索空间边界，我们喜欢在我们称为基本超参数轴图上绘制完成的试验，其中我们绘制验证目标值与其中一个超参数(例如学习率)的关系。图上的每个点都对应一次试验。

    - 每个trail的验证目标值通常应该是在训练过程中获得的最优结果。

<p align="center" id="figure-1">
<img src="assets/bad_search_space.png" width="49%" alt="Example of bad search space boundaries">
<img src="assets/good_search_space.png" width="49%" alt="Example of good search space boundaries">
</p>

<p align="center"><b>Figure 1:</b> 一个坏的搜索范围和一个良好的搜索范围的比较.左边的最优参数出现在了搜索范围的边界处，说明该搜索边界不是最优的。</p>

- 图1中的图表显示了错误率(越低越好)与初始学习率的关系。
- 如果最佳点聚集在搜索空间的边缘(在某些维度上)，那么搜索空间边界可能需要扩展，直到最佳观察点不再靠近边界。
- 通常，一项研究将包括“不可行”的试验，这些试验偏离或得到非常糟糕的结果(在上面的图中用红色x标记)。
    - 如果所有的试验对于学习率大于某个阈值都是不可可行的，并且如果表现最好的试验在该区域的边缘有学习率，那么模型可能会受到稳定性问题的影响，从而无法获得更高的学习率。

</details>

#### 判断在搜索空间中是否采样了足够的点

<details><summary><em>[Click to expand]</em></summary>

<br>

- 一般来说，很难知道搜索空间的采样是否足够密集。🤖
- 进行更多的试验当然更好，但代价很明显。
- 因为很难知道我们什么时候已经采样了足够多，我们通常会采样我们能承受的范围，并试图通过反复查看各种超参数轴图来校准我们的直觉信心，并试图获得搜索空间的“好”区域中有多少点。


</details>

#### 检查训练曲线



***总结:*** *检查训练曲线是识别常见故障模式的简单方法，可以帮助我们优先考虑下一步要采取的行动。*

- 虽然在许多情况下，我们实验的主要目标只需要考虑每次试验的验证误差，但在将每次试验减少到单个数字时，我们必须小心，因为它可能隐藏了表面之下发生的事情的重要细节。
- 对于每一项研究，我们总是查看至少最好的几个试验的训练曲线(训练误差和验证误差与训练持续时间内的训练步骤的关系)。
- 即使这不是解决主要实验目标所必需的，检查训练曲线是确定常见故障模式的简单方法，并可以帮助我们优先考虑下一步采取的行动。
- 在检查训练曲线时，我们对以下问题感兴趣。
- 是否有任何试验显示出有过拟合的问题?
    - 当验证错误在训练过程中的某个点开始增加时，就会出现问题过拟合。
    - 在实验设置中，我们通过为每个科学超参数设置选择“最佳”试验来优化干扰超参，我们应该检查至少每个与我们正在比较的科学超参数设置对应的最佳试验中是否有问题过拟合。
        - 如果任何一个最佳试验显示出有问题的过拟合，我们通常希望在比较科学超参数的值之前，使用额外的正则化技术重新运行实验和/或更好地调整现有的正则化参数。
            - 如果科学超参数包括正则化参数，这可能不适用，因为如果这些正则化参数的低强度设置导致有过拟合问题也就不足为奇了。
    - 减少过拟合通常是简单的，使用常见的正则化技术，添加最小的代码复杂性或额外的计算(例如，dropout，标签平滑，权重衰减)，所以在下一轮实验中添加一个或多个这样的技术通常不是什么大问题。
    - 例如，如果科学超参数是“隐藏层数”，而使用最大隐藏层数的最佳试验表现出有问题的过拟合，那么我们通常更喜欢用额外的正则化再次尝试，而不是立即选择较小数量的隐藏层。
    - 即使没有一个“最佳”试验表现出过拟合，如果它发生在任何一个试验中，仍然可能是一个问题。
        - 选择最佳的参数抑制了表现出有问题的过拟合的配置，并有利于那些没有过拟合的配置。换句话说，它将倾向于更正则化的配置。
        - 然而，任何使训练变得更糟的东西都可以作为正则化因子，即使它不是故意的。例如，选择较小的学习率可以通过阻碍优化过程来正则化训练，但我们通常不希望以这种方式选择学习率。
        - 因此，我们必须意识到，科学超参数的每一种设置的“最佳”试验可能导致某些科学超参数或干扰超参选到“坏”值。
- 在训练中是否存在错误率变化较大的问题（loss曲线不平滑）?
    - 如果是这样，这可能会干扰我们比较不同科学超参数值的能力(因为每次试验随机地结束于“幸运”或“不幸”步骤)，以及我们复现最佳试验结果的能力(因为生产模型可能不会像研究中那样结束于相同的“幸运”步骤)。
    - 最可能导致步进方差（loss不平滑）的原因是Batch数据方差较大(因为每个batch是随机抽样本)、验证集过小，以及在训练后期使用过高的学习率。
    - 可能的补救措施包括增加Batchsize大小、获得更多验证数据、使用学习率衰减或使用Polyak平均。
- 训练结束后，试验是否仍在改善?
    -  如果是这样，这表明我们处于“计算约束”状态，我们可能会从增加训练步骤的数量或改变学习率计划中受益。
- 在最后的训练步骤之前，训练和验证集的性能已经饱和了吗?
    - 如果是这样，这表明我们处于“不受计算限制”的状态，并且我们可能能够减少训练步骤的数量。
- 虽然我们无法一一列举，但通过检查训练曲线，可以明显地发现许多其他的行为(例如，训练过程中训练损失的增加通常表明训练管道中存在错误)。
 

</details>

#### 使用隔离图检测更改是否有用


<details><summary><em>[Click to expand]</em></summary>

<br>


<p align="center" id="figure-2">
<img src="assets/isolation_plot.png" width="49%" alt="Isolation plot that investigates the best value of weight decay for ResNet-50
trained on ImageNet.">
</p>

<p align="center"><b>Figure 2:</b>研究在ImageNet上训练的ResNet-50的权重衰减的最佳值的隔离图。</p>

- 通常，一组实验的目标是比较科学超参数的不同值。
    - 例如，我们可能想要确定导致最佳验证错误的权重衰减值。
- 隔离图是基本超参数轴图的一种特殊情况。隔离图上的每个点对应于在一些(或全部)干扰超参上的最佳试验的性能。
    - 换句话说，我们在“优化掉”干扰超参后绘制模型性能。
- 使用隔离图可以更容易地在科学超参数的不同值之间进行比较。
- 例如，图2显示了对于在ImageNet上训练的ResNet-50的特定配置产生最佳验证性能的权重衰减值（weight decay）。
    - 如果我们的目标是确定是否包含权重衰减，那么我们将比较该图中的最佳点与没有权重衰减的基线。为了进行公平的比较，基线还应该对其学习率进行同样良好的调整。
- 当我们拥有由(准)随机搜索生成的数据并考虑隔离图的连续超参数时，我们可以通过将基本超参数轴图的x轴值分桶并在每个桶切片中选择最优结果来近似隔离图。

</details>

#### Automate generically useful plots

<details><summary><em>[Click to expand]</em></summary>

<br>

- 生成图表所花费的精力越多，我们就可能很难尽可能多得查看它们，所以我们有必要设置基础设施来自动生成尽可能多的图。
- 至少，我们会自动为我们在实验中改变的所有超参数生成基本超参数轴图。
- 此外，我们自动生成所有试验的训练曲线，并尽可能容易地找到每个研究中最好的几个试验，并检查它们的训练曲线。
- 我们还可以添加许多其他有用的潜在图形和可视化。虽然上面描述的是一个很好的起点，但套用Geoffrey Hinton的话，“每次你画出一些新的图形，你就会学到一些新东西。”

</details>

### Determining whether to adopt a training pipeline change or hyperparameter configuration

***Summary:*** *When deciding whether to make a change to our model or training
procedure or adopt a new hyperparameter configuration going forward, we need to
be aware of the different sources of variation in our results.*

-   When we are trying to improve our model, we might observe that a particular
    candidate change initially achieves a better validation error compared to
    our incumbent configuration, but find that after repeating the experiment
    there is no consistent advantage. Informally, we can group the most
    important sources of variation that might cause such an inconsistent result
    into the following broad categories:
    -   **Training procedure variance**, **retrain variance**, or **trial
        variance**: the variation we see between training runs that use the same
        hyperparameters, but different random seeds.
        -   For example, different random initializations, training data
            shuffles, dropout masks, patterns of data augmentation operations,
            and orderings of parallel arithmetic operations, are all potential
            sources of trial variance.
    -   **Hyperparameter search variance**, or **study variance**: the variation
        in results caused by our procedure to select the hyperparameters.
        -   For example, we might run the same experiment with a particular
            search space, but with two different seeds for quasi-random search
            and end up selecting different hyperparameter values.
    -   **Data collection and sampling variance**: the variance from any sort of
        random split into training, validation, and test data or variance due to
        the training data generation process more generally.
-   It is all well and good to make comparisons of validation error rates
    estimated on a finite validation set using fastidious statistical tests, but
    often the trial variance alone can produce statistically significant
    differences between two different trained models that use the same
    hyperparameter settings.
-   We are most concerned about study variance when trying to make conclusions
    that go beyond the level of an individual point in hyperparameters space.
    -   The study variance depends on the number of trials and the search space
        and we have seen cases where it is larger than the trial variance as
        well as cases where it is much smaller.
-   Therefore, before adopting a candidate change, consider running the best
    trial N times to characterize the run-to-run trial variance.
    -   Usually, we can get away with only recharacterizing the trial variance
        after major changes to the pipeline, but in some applications we might
        need fresher estimates.
    -   In other applications, characterizing the trial variance is too costly
        to be worth it.
-   At the end of the day, although we only want to adopt changes (including new
    hyperparameter configurations) that produce real improvements, demanding
    complete certainty that something helps isn't the right answer either.
-   Therefore, if a new hyperparameter point (or other change) gets a better
    result than the baseline (taking into account the retrain variance of both
    the new point and the baseline as best we can), then we probably should
    adopt it as the new baseline for future comparisons.
    -   However, we should only adopt changes that produce improvements that
        outweigh any complexity they add.

### After exploration concludes

***Summary:*** *Bayesian optimization tools are a compelling option once we’re
done exploring for good search spaces and have decided what hyperparameters even
should be tuned at all.*

-   At some point, our priorities will shift from learning more about the tuning
    problem to producing a single best configuration to launch or otherwise use.
-   At this point, there should be a refined search space that comfortably
    contains the local region around the best observed trial and has been
    adequately sampled.
-   Our exploration work should have revealed the most essential hyperparameters
    to tune (as well as sensible ranges for them) that we can use to construct a
    search space for a final automated tuning study using as large a tuning
    budget as possible.
-   Since we no longer care about maximizing our insight into the tuning
    problem, many of
    [the advantages of quasi-random search](#why-use-quasi-random-search-instead-of-more-sophisticated-black-box-optimization-algorithms-during-the-exploration-phase-of-tuning)
    no longer apply and Bayesian optimization tools should be used to
    automatically find the best hyperparameter configuration.
    -   If the search space contains a non-trivial volume of divergent points
        (points that get NaN training loss or even training loss many standard
        deviations worse than the mean), it is important to use black box
        optimization tools that properly handle trials that diverge (see
        [Bayesian Optimization with Unknown Constraints](https://arxiv.org/abs/1403.5607)
        for an excellent way to deal with this issue).
-   At this point, we should also consider checking the performance on the test
    set.
    -   In principle, we could even fold the validation set into the training
        set and retraining the best configuration found with Bayesian
        optimization. However, this is only appropriate if there won't be future
        launches with this specific workload (e.g. a one-time Kaggle
        competition).

## Determining the number of steps for each training run

-   There are two types of workloads: those that are compute-bound and those
    that are not.
-   When training is **compute-bound**, training is limited by how long we are
    willing to wait and not by how much training data we have or some other
    factor.
    -   In this case, if we can somehow train longer or more efficiently, we
        should see a lower training loss and, with proper tuning, an improved
        validation loss.
    -   In other words, *speeding up* training is equivalent to *improving*
        training and the "optimal" training time is always "as long as we can
        afford."
    -   That said, just because a workload is compute-limited doesn't mean
        training longer/faster is the only way to improve results.
-   When training is **not compute-bound**, we can afford to train as long as we
    would like to, and, at some point, training longer doesn't help much (or
    even causes problematic overfitting).
    -   In this case, we should expect to be able to train to very low training
        loss, to the point where training longer might slightly reduce the
        training loss, but will not meaningfully reduce the validation loss.
    -   Particularly when training is not compute-bound, a more generous
        training time budget can make tuning easier, especially when tuning
        learning rate decay schedules, since they have a particularly strong
        interaction with the training budget.
        -   In other words, very stingy training time budgets might require a
            learning rate decay schedule tuned to perfection in order to achieve
            a good error rate.
-   Regardless of whether a given workload is compute-bound or not, methods that
    increase the variance of the gradients (across batches) will usually result
    in slower training progress, and thus may increase the number of training
    steps required to reach a particular validation loss. High gradient variance
    can be caused by:
    -   Using a smaller batch size
    -   Adding data augmentation
    -   Adding some types of regularization (e.g. dropout)

### Deciding how long to train when training is *not* compute-bound

-   Our main goal is to ensure we are training long enough for the model to
    reach the best possible result, while avoiding being overly wasteful in the
    number of training steps.
-   When in doubt, err on the side of training longer. Performance should never
    degrade when training longer, assuming retrospective (optimal) checkpoint
    selection is used properly and checkpoints are frequent enough.
-   Never tune the `max_train_steps` number in a study. Pick a value and use it
    for all trials. From these trials, plot the training step that retrospective
    checkpoint selection finds in order to refine the choice of
    `max_train_steps`.
    -   For example, if the best step is always during the first 10% of
        training, then the maximum number of steps is way too high.
    -   Alternatively, if the best step is consistently in the last 25% of
        training we might benefit from training longer and re-tuning the decay
        schedule.
-   The ideal number of training steps can change when the architecture or data
    changes (e.g. adding data augmentation).
-   Below we describe how to pick an initial candidate value for
    `max_train_steps` based on the number of steps necessary to "perfectly fit"
    the training set using a constant learning rate.
    -   Note, we are not using the phrase "perfectly fit the training set" in a
        precise or mathematically well-defined way. It is merely meant as an
        informal descriptor to indicate a very low training loss.
        -   For example, when training with the log loss, absent regularization
            terms, we might see the training loss keep slowly improving until we
            reach floating point limits as the network weights grow without
            bound and the predictions of the model on the training set become
            increasingly confident. In this case, we might say the model
            "perfectly fit" the training set around the time the
            misclassification error reached zero on the training set.
    -   The starting value for `max_train_steps` we find may need to be
        increased if the amount of gradient noise in the training procedure
        increases.
        -   For example, if data augmentation or regularizers like dropout are
            introduced to the model.
    -   It may be possible to decrease `max_train_steps` if the training process
        improves somehow.
        -   For example, with a better tuned optimizer or a better tuned
            learning rate schedule.

#### Algorithm for picking an initial candidate for max_train_steps using a learning rate sweep

<details><summary><em>[Click to expand]</em></summary>

<br>

-   This procedure assumes it is possible to not only "perfectly" fit the
    training set, but to do so using a constant learning rate schedule.
-   If it is possible to perfectly fit the entire training set, then there must
    exist a configuration (with some value of `max_train_steps`) that perfectly
    fits the training set; find any such configuration and use its value of
    `max_train_steps` as a starting point `N`.
-   Run a constant learning rate sweep (i.e. grid search the learning rate)
    without data augmentation and without regularization where each trial trains
    for `N` steps.
-   The number of steps required for the fastest trial in the sweep to reach
    perfect training performance is our initial guess for `max_train_steps`.
-   **NOTE:** Bad search spaces can make it possible to engage in
    self-deception.
    -   For example, if all the learning rates in a study are too small, we
        might incorrectly conclude that a very large value of `max_train_steps`
        is necessary.
    -   At a minimum, we should check that the optimal learning rate in the
        study is not at the boundary of the search space.

</details>

### Deciding how long to train when training is compute-bound

-   In some cases, training loss keeps improving indefinitely and our patience
    and computational resources become the limiting factors.
-   If training loss (or even validation loss) keeps improving indefinitely,
    should we always train as long as we can afford? Not necessarily.
    -   We might be able to tune more effectively by running a larger number of
        shorter experiments and reserving the longest "production length" runs
        for the models we hope to launch.
    -   As the training time for trials approaches our patience limit, tuning
        experiments become more relevant for our potential launch candidates,
        but we can complete fewer of them.
    -   There are probably many questions we can answer while only training for
        ~10% of the production length, but there is always a risk that our
        conclusions at this time limit will not apply to experiments at 20% of
        the production length, let alone 100%.
-   Tuning in multiple rounds with increasing, per-trial training step limits is
    a sensible approach.
    -   We can do as many rounds as we want, but usually 1-3 are the most
        practical.
    -   Essentially, try to obtain as much understanding of the problem as
        possible using trials with a very quick turnaround time, trading off
        tuning thoroughness with relevance to the final, longest runs.
    -   Once a given per-trial time limit has generated useful insights, we can
        increase the training time and continue tuning, double-checking our
        conclusions from the shorter runs as needed.
-   As a starting point, we recommend two rounds of tuning:
    -   Round 1: Shorter runs to find good model and optimizer hyperparameters.
    -   Round 2: Very few long runs on good hyperparameter points to get the
        final model.
-   The biggest question going from `Round i` &rarr; `Round i+1` is how to
    adjust learning rate decay schedules.
    -   One common pitfall when adjusting learning rate schedules between rounds
        is using all the extra training steps with too small of a learning rate.

#### Round 1

<details><summary><em>[Click to expand]</em></summary>

<br>

-   Unfortunately, there is no guarantee that good hyperparameters found in
    short, incomplete training are still good choices when training length is
    significantly increased. However, for some kinds of hyperparameters, they
    are often correlated enough for Round 1 to be useful.
-   What hyperparameter values found in shorter runs do we expect to transfer to
    longer training runs? For all of this, we need more research. But based on
    what we know so far, here are the authors’ suspicions in order of decreasing
    probability of transferring:
    -   Very likely to transfer
        -   Early training instability can be resolved in the first round of
            tuning using a smaller number of training steps. Perhaps these
            hyperparameters are the closest thing to a sure bet for transfer
            that we have.
            -   Warmup length
            -   Initialization
    -   Likely to transfer
        -   Model architecture - A dramatic win in the model architecture will
            usually transfer, but there are probably many counterexamples.
    -   Might transfer
        -   Optimization algorithm/optimizer hyperparameters - We think this
            would "loosely" transfer. It’s definitely weaker than the things
            above it.
        -   Data augmentation
        -   Regularization
            -   If it isn't possible to perfectly fit the training set, the
                model might be in a regime where regularization is unlikely to
                help very much.
    -   Unlikely to transfer
        -   Learning rate schedule: unlikely to transfer perfectly.
            -   [This paper](https://arxiv.org/abs/2203.15556) suggests that
                even decay schedule transfers, but we don't believe this is true
                in general. Example: Tuning sqrt decay on small # of training
                steps then extending to large # will result in the majority of
                training occurring at overly small steps.
                -   One can likely do "good enough" with most schedules in the
                    limit of extreme training budget, but noticeable performance
                    improvements can likely be seen if it is tuned.
            -   [Understanding Short-Horizon Bias in Stochastic
                Meta-Optimization](https://arxiv.org/abs/1803.02021) describes
                the dangers of trying to pick learning rates myopically.

</details>

#### Round 2

<details><summary><em>[Click to expand]</em></summary>

<br>

-   Run the best hyperparameter configuration from Round 1.
-   **(Speculation)** 🤖 Use the extra steps to extend the period of training at
    a high learning rate.
    -   E.g. if linear schedule then keep the length of the decay fixed from
        Round 1 and extend the period of constant lr in the beginning.
    -   For cosine decay, just keep the base lr from Round 1 and extend
        `max_train_steps` as in
        [Chinchilla paper](https://arxiv.org/abs/2203.15556).
-   More rounds might make sense for teams with very mature modeling and tuning
    pipelines and very long and expensive production training runs, but they
    will often be overkill.
    -   We've described how to transfer from Step 1 &rarr; Step 2. If we didn't care
        about analysis time and if making efficient use of compute was the
        overriding concern, then the ideal would be to exponentially increase
        the length of training runs (and thus the end-to-end time to complete a
        study) over many different rounds of tuning.
        -   At each round we systematically ensure our choices continue to hold
            up.
        -   New ideas go through a pipeline that progressively derisks them
            using increasingly long-running experiments from Step i to Step i+1.

</details>

## Additional guidance for the training pipeline

### Optimizing the input pipeline

***Summary:*** *The causes and interventions of input-bound pipelines are highly
task-dependent; use a profiler and look out for common issues.*

-   Use an appropriate profiler to diagnose input-bound pipelines. For example,
    [Perfetto](https://jax.readthedocs.io/en/latest/profiling.html) for JAX or
    [TensorFlow profiler](https://www.tensorflow.org/guide/profiler) for
    TensorFlow.
-   Ultimately, the specific causes and interventions will be highly
    task-dependent. Broader engineering considerations (e.g. minimizing disk
    footprint) may warrant worse input pipeline performance.
-   Common causes:
    -   Data are not colocated with the training process, causing I/O latency
        (this might happen when reading training data over a network).
    -   Expensive online data preprocessing (consider doing this once offline
        and saving).
    -   Unintentional synchronization barriers that interfere with data pipeline
        prefetching. For example, when synchronizing metrics between the device
        and host in CommonLoopUtils
        ([link](https://github.com/google/CommonLoopUtils/blob/fea2518ada8814a78e1492023fd9f00edb0b0568/clu/metrics.py#L291)).
-   Common tips:
    -   Instrument input pipeline to prefetch examples (e.g.
        [tf.data.Dataset.prefetch](https://www.tensorflow.org/guide/data_performance#prefetching))
    -   Remove unused features/metadata from each as early in the pipeline as
        possible.
    -   Increase the replication of the number of jobs generating examples for
        the input pipeline. For example, by using the
        [tf.data service](https://www.tensorflow.org/api_docs/python/tf/data/experimental/service).

### Evaluating model performance

***Summary:*** *Run evaluation at larger batch sizes than training. Run
evaluations at regular step intervals, not regular time intervals.*

#### Evaluation settings

<details><summary><em>[Click to expand]</em></summary>

<br>

-   There are several settings in which we can evaluate the performance of our
    models.
    -   **Online evaluation** - metrics are collected when the model is serving
        predictions in a production environment.
    -   **Offline evaluation** - metrics are collected when the model is run on
        offline train/validation/test sets that are representative of the
        production environment.
    -   **Periodic evaluations** - metrics are collected during model training
        that might either be a proxy for the offline evaluation, and/or on a
        subset of the data used in offline evaluation.
-   Online evaluation is the gold standard, but is often impractical during the
    model development phase.
-   Depending on the problem, offline evaluation can be fairly involved and
    computationally expensive.
-   Periodic evaluations are the most practical and economical choice, but may
    not fully represent the production environment.
    -   Our goal during periodic evaluation is to use an expedient proxy of the
        offline evaluation, without sacrificing the reliability of the signal we
        get during training.

</details>

#### Setting up periodic evaluations

<details><summary><em>[Click to expand]</em></summary>

<br>

-   We run periodic evaluations during training to monitor its progress in real
    time, to
    [facilitate retrospective model checkpoint selection](#saving-checkpoints-and-retrospectively-selecting-the-best-checkpoint),
    and so that we can
    [examine the training curves at the end of training](#examining-the-training-curves).
-   The simplest configuration is to perform both training and periodic
    evaluations within the same compute instance, periodically alternating
    between training and evaluation.
    -   In this case, the batch size used to perform evaluations should be *at
        least* as large as the batch size used for training because model
        activations don't need to be maintained during evaluation, lowering the
        computational requirements per example.
-   Periodic evaluations should be done at regular step intervals, not time
    intervals.
    -   Evaluating based on time intervals can make it harder to interpret the
        training curves, especially when training may suffer from preemptions of
        the training jobs, network latency issues, etc.
-   Periodicity in valid/test metrics (when using a shuffled
    train/validation/test split) can indicate implementation bugs such as test
    data having overlap with training data, or training data not being properly
    shuffled. Evaluating at regular step intervals can make these issues easier
    to catch.
-   Partial batches can occur when the evaluation sets are not divisible by the
    batch size. Ensure that the padded examples are correctly weighed to prevent
    the loss function from being biased by them. Often, these padded examples
    can be given a weight of zero.
-   Save sufficient information per evaluation to support offline analysis.
    Ideally, we would save predictions on a selection of individual examples
    since they can be invaluable for debugging.
    -   Generating artifacts like
        [SavedModels](https://www.tensorflow.org/guide/saved_model) make it easy
        to do ad-hoc model inspection after evaluation jobs finish.

</details>

#### Choosing a sample for periodic evaluation

<details><summary><em>[Click to expand]</em></summary>

<br>

-   The periodic evaluation job might not run fast enough to compute metrics on
    the full offline evaluation set in a reasonable amount of time. This often
    necessitates sampling data for periodic evaluation.
-   We consider the following factors when constructing a sampled dataset:
    -   <ins>Sample size</ins>
        -   Check that the performance computed on the sampled dataset used by
            the periodic job matches the performance on the whole offline
            evaluation set, i.e. there is no skew between the sampled set and
            the full dataset.
        -   The dataset used for periodic evaluation should be small enough that
            it’s easy to generate model predictions over its entirety, but large
            enough that improvements to the model can be accurately measured
            (i.e. not overwhelmed by label noise).
        -   It should be large enough to accommodate multiple such evaluations
            across trials in sequence, and still produce accurate estimates.
            That is, to avoid adaptively "fitting" to the validation set over
            time, in a way that doesn't generalize to a held-out test set.
            However, this consideration is rarely a practical concern.
    -   <ins>Imbalanced datasets</ins>
        -   For imbalanced datasets, performance on rare classes of examples
            will often be noisy.
        -   For datasets with a small number of examples in a class label, log
            the number of examples predicted correctly to get more insight into
            accuracy improvements (.05 sensitivity improvement sounds exciting,
            but was it just one more example correct?).

</details>

### Saving checkpoints and retrospectively selecting the best checkpoint

***Summary:*** *Run training for a fixed number of steps and retrospectively
choose the best checkpoint from the run.*

-   Most deep learning frameworks support
    [model checkpointing](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html).
    That is, the current state of the model is periodically preserved on disk.
    This allows the training job to be resilient to compute instance
    interruptions.
-   The best checkpoint is often not the last checkpoint, particularly when the
    validation set performance does not continue to increase over time but
    rather fluctuates about a particular value.
-   Set up the pipeline to keep track of the N best checkpoints seen so far
    during training. At the end of training, model selection is then a matter of
    choosing the best checkpoint seen during training. We call this
    **retrospective optimal checkpoint selection**.
-   Supporting prospective early stopping is usually not necessary, since we’re
    pre-specifying a trial budget and are preserving the N best checkpoints seen
    so far.

### Setting up experiment tracking

***Summary:*** *When tracking different experiments, make sure to note a number
of essentials like the best performance of a checkpoint in the study, and a
short description of the study.*

-   We've found that keeping track of experiment results in a spreadsheet has
    been helpful for the sorts of modeling problems we've worked on. It often
    has the following columns:
    -   Study name
    -   A link to wherever the config for the study is stored.
    -   Notes or a short description of the study.
    -   Number of trials run
    -   Performance on the validation set of the best checkpoint in the study.
    -   Specific reproduction commands or notes on what unsubmitted changes were
        necessary to launch training.
-   Find a tracking system that captures at least the information listed above
    and is convenient for the people doing it. Untracked experiments might as
    well not exist.

### Batch normalization implementation details

***Summary:*** *Nowadays batch norm can often be replaced with LayerNorm, but in
cases where it cannot, there are tricky details when changing the batch size or
number of hosts.*

-   Batch norm normalizes activations using their mean and variance over the
    current batch, but in the multi-device setting these statistics are
    different on each device unless explicitly synchronized.
-   Anecdotal reports (mostly on ImageNet) say calculating these normalizing
    statistics using only ~64 examples actually works better in practice (see
    Ghost Batch Norm from [this paper](https://arxiv.org/abs/1705.08741)).
-   Decoupling the total batch size and the number of examples used to calculate
    batch norm statistics is particularly useful for batch size comparisons.
-   Ghost batch norm implementations do not always correctly handle the case
    where the per-device batch size > virtual batch size. In this case we'd
    actually need to subsample the batch on each device in order to get the
    proper number of batch norm statistic examples.
-   Exponential moving averages used in test mode batch norm are just a linear
    combination of training statistics, so these EMAs only need to be
    synchronized before saving them in checkpoints. However, some common
    implementations of batch norm do not synchronize these EMAs and only save
    the EMA from the first device.

### Considerations for multi-host pipelines

***Summary:*** *for logging, evals, RNGs, checkpointing, and data sharding,
multi-host training can make it very easy to introduce bugs!*

-   Ensure the pipeline is only logging and checkpointing on one host.
-   Make sure before evaluation or checkpointing is run, the batch norm
    statistics are synchronized across hosts.
-   It is critical to have RNG seeds that are the same across hosts (for model
    initialization), and seeds that are different across hosts (for data
    shuffling/preprocessing), so make sure to mark them appropriately.
-   Sharding data files across hosts is usually recommended for improved
    performance.

## FAQs

### What is the best learning rate decay schedule family?

<details><summary><em>[Click to expand]</em></summary>

<br>

-   It’s an open problem. It’s not clear how to construct a set of rigorous
    experiments to confidently answer what the "best" LR decay schedule is.
-   Although we don't know the best schedule family, we're confident that it’s
    important to have some (non-constant) schedule and that tuning it matters.
-   Different learning rates work best at different times during the
    optimization process. Having some sort of schedule makes it more likely for
    the model to hit a good learning rate.

</details>

### Which learning rate decay should I use as a default?

<details><summary><em>[Click to expand]</em></summary>
<br>

-   Our preference is either linear decay or cosine decay, and a bunch of other
    schedule families are probably good too.

</details>

### Why do some papers have complicated learning rate schedules?

<details><summary><em>[Click to expand]</em></summary>
<br>

-   It’s not uncommon to see papers with complicated piecewise learning rate
    (LR) decay schedules.
-   Readers often wonder how the authors arrived at such a complicated study.
-   Many complicated LR decay schedules are the result of tuning the schedule as
    a function of the validation set performance in an ad hoc way:
    1.  Start a single training run with some simple LR decay (or a constant
        learning rate).
    2.  Keep training running until the performance seems to stagnate. If this
        happens, pause training. Resume it with a perhaps steeper LR decay
        schedule (or smaller constant learning rate) from this point. Repeat
        this process until the conference/launch deadline.
-   Blithely copying the resulting *schedule* is generally not a good idea since
    the best particular schedule will be sensitive to a host of other
    hyperparameter choices.
    -   Better to copy the *algorithm* that produced the schedule, although this
        is rarely possible when arbitrary human judgment produced the schedule.
-   This type of validation-error-sensitive schedule is fine to use if it can be
    fully automated, but human-in-the-loop schedules that are a function of
    validation error are brittle and not easily reproducible, so we recommend
    avoiding them.
    -   Before publishing results that used such a schedule, please try to make
        it fully reproducible.

</details>

### How should Adam’s hyperparameters be tuned?

<details><summary><em>[Click to expand]</em></summary>
<br>

-   As discussed above, making general statements about search spaces and how
    many points one should sample from the search space is very difficult. Note
    that not all the hyperparameters in Adam are equally important. The
    following rules of thumb correspond to different "budgets" for the number of
    trials in a study.
    -   If < 10 trials in a study, only tune the (base) learning rate.
    -   If 10-25 trials, tune learning rate and $\beta_1$.
    -   If 25+ trials, tune the learning rate, $\beta_1$ and $\epsilon$.
    -   If one can run substantially more than 25 trials, additionally tune
        $\beta_2$.

</details>

### Why use quasi-random search instead of more sophisticated black box optimization algorithms during the exploration phase of tuning?

<details><summary><em>[Click to expand]</em></summary>

-   Quasi-random search (based on
    [low-discrepancy sequences](https://en.wikipedia.org/wiki/Low-discrepancy_sequence))
    is our preference over fancier black box optimization tools when used as
    part of an iterative tuning process intended to maximize insight into the
    tuning problem (what we refer to as the "exploration phase"). Bayesian
    optimization and similar tools are more appropriate for the exploitation
    phase.
-   Quasi-random search based on randomly shifted low-discrepancy sequences can
    be thought of as "jittered, shuffled grid search", since it uniformly, but
    randomly, explores a given search space and spreads out the search points
    more than random search.
-   The advantages of quasi-random search over more sophisticated black box
    optimization tools (e.g. Bayesian optimization, evolutionary algorithms)
    include:
    1.  Sampling the search space non-adaptively makes it possible to change the
        tuning objective in post hoc analysis without rerunning experiments.
        -   For example, we usually want to find the best trial in terms of
            validation error achieved at any point in training. But the
            non-adaptive nature of quasi-random search makes it possible to find
            the best trial based on final validation error, training error, or
            some alternative evaluation metric without rerunning any
            experiments.
    2.  Quasi-random search behaves in a consistent and statistically
        reproducible way.
        -   It should be possible to reproduce a study from six months ago even
            if the implementation of the search algorithm changes, as long as it
            maintains the same uniformity properties. If using sophisticated
            Bayesian optimization software, the implementation might change in
            an important way between versions, making it much harder to
            reproduce an old search. It isn’t always possible to roll back to an
            old implementation (e.g. if the optimization tool is run as a
            service).
    3.  Its uniform exploration of the search space makes it easier to reason
        about the results and what they might suggest about the search space.
        -   For example, if the best point in the traversal of quasi-random
            search is at the boundary of the search space, this is a good (but
            not foolproof) signal that the search space bounds should be
            changed. [This section](#identifying-bad-search-space-boundaries)
            goes into more depth. However, an adaptive black box optimization
            algorithm might have neglected the middle of the search space
            because of some unlucky early trials even if it happens to contain
            equally good points, since it is this exact sort of non-uniformity
            that a good optimization algorithm needs to employ to speed up the
            search.
    4.  Running different numbers of trials in parallel versus sequentially will
        not produce statistically different results when using quasi-random
        search (or other non-adaptive search algorithms), unlike with adaptive
        algorithms.
    5.  More sophisticated search algorithms may not always handle infeasible
        points correctly, especially if they aren't designed with neural network
        hyperparameter tuning in mind.
    6.  Quasi-random search is simple and works especially well when many tuning
        trials will be running in parallel.
        -   Anecdotally[^3], it is very hard for an adaptive algorithm to beat a
            quasi-random search that has 2X its budget, especially when many
            trials need to be run in parallel (and thus there are very few
            chances to make use of previous trial results when launching new
            trials).
        -   Without expertise in Bayesian optimization and other advanced black
            box optimization methods, we might not achieve the benefits they
            are, in principle, capable of providing. It is hard to benchmark
            advanced black box optimization algorithms in realistic deep
            learning tuning conditions. They are a very active area of current
            research, and the more sophisticated algorithms come with their own
            pitfalls for inexperienced users. Experts in these methods are able
            to get good results, but in high-parallelism conditions the search
            space and budget tend to matter a lot more.
-   That said, if our computational resources only allow a small number of
    trials to run in parallel and we can afford to run many trials in sequence,
    Bayesian optimization becomes much more attractive despite making our tuning
    results harder to interpret.

[^3]: Ben Recht and Kevin Jamieson
    [pointed out](http://www.argmin.net/2016/06/20/hypertuning/) how strong
    2X-budget random search is as a baseline (the
    [Hyperband paper](https://jmlr.org/papers/volume18/16-558/16-558.pdf)
    makes similar arguments), but it is certainly possible to find search
    spaces and problems where state-of-the-art Bayesian optimization
    techniques crush random search that has 2X the budget. However, in our
    experience beating 2X-budget random search gets much harder in the
    high-parallelism regime since Bayesian optimization has no opportunity to
    observe the results of previous trials.

</details>

### Where can I find an implementation of quasi-random search?

<details><summary><em>[Click to expand]</em></summary>
<br>

-   We use
    [this implementation](https://github.com/mlcommons/algorithmic-efficiency/blob/main/algorithmic_efficiency/halton.py)
    that generates a Halton sequence for a given search space (intended to
    implement a shifted, scrambled Halton sequence as recommended in
    https://arxiv.org/abs/1706.03200).
-   If a quasi-random search algorithm based on a low-discrepancy sequence is
    not available, it is possible to substitute pseudo random uniform search
    instead, although this is likely to be slightly less efficient.
    -   In 1-2 dimensions, grid search is also acceptable, although not in
        higher dimensions (see
        [Bergstra & Bengio, 2012](https://www.jmlr.org/papers/v13/bergstra12a.html)).

</details>

### How many trials are needed to get good results with quasi-random search?

<details><summary><em>[Click to expand]</em></summary>
<br>

<p align="center">
<img src="assets/have_we_sampled_enough.png" width="49%" alt="A box plot showing the importance of sampling enough">
</p>

<p align="center"><b>Figure 3:</b> A ResNet-50 was tuned on ImageNet with 100
trials. Via bootstrapping, different amounts of tuning budget were simulated.
Box plots of the best performances for each trial budget are plotted above.

-   There is no way to answer this question in general, but we can look at
    specific examples.
-   As the Figure 3 shows, the number of trials in a study can have a
    substantial impact on the results.
    -   Notice how large the interquartile ranges are when 6 trials were
        sampled, versus when 20 trials were sampled.
    -   Even with 20 trials, it is likely that the difference between especially
        lucky and unlucky studies will be larger than the typical variation
        between re-trains of this model on different random seeds, with fixed
        hyperparameters, which for this workload might be around +/- 0.1% on a
        validation error rate of \~23%.

</details>

### How can optimization failures be debugged and mitigated?

<details><summary><em>[Click to expand]</em></summary>
<br>


***Summary:*** *If the model is experiencing optimization difficulties, it’s
important to fix them before trying other things. Diagnosing and correcting
training failures is an active area of research.*

<p align="center">
<img src="assets/stride_instability.png" width="80%" alt="Changing the strides in a single residual block in a WideResnet results in training instability.">
</p>


<p align="center"><b>Figure 4:</b> Changing the strides in a single residual block (2x2 -> 1x1) in a WideResnet results in training instability. This does not degrade performance at low learning rates, but high learning rates no longer train well due to the instability. Applying 1000 steps of learning rate warmup resolves this particular instance of instability, allowing stable training at max learning rate of .1.</p>

#### Identifying unstable workloads

-   Any workload will become unstable if the learning rate is too large.
    Instability is only an issue when it forces us to use a learning rate that’s
    too small.
-   There are at least two types of training instability worth distinguishing:
    1.  Instability at initialization/early in training.
    2.  Sudden instability in the middle of training.
-   We can take a systematic approach to identifying stability issues in our
    workload.
    1.  Do a learning rate sweep and find the best learning rate lr*.
    2.  Plot training loss curves for learning rates just above lr*.
    3.  If the learning rates > lr* show loss instability (loss goes up not down
        during periods of training), then it is likely that fixing the
        instability will result in better training.
-   Log the L2 norm of the full loss gradient during training, outlier values
    can result in spurious instability in the middle of training. This can
    inform how to pick gradient/update clipping.

**NOTE:** Some models show very early instability followed by a recovery that
results in slow but stable training. **Common evaluation schedules can miss
these issues by not evaluating frequently enough!**

To check for this, we can train for an abbreviated run of just \~500 steps using
`lr = 2 * current best`, but evaluate every step.

<p align="center">
<img src="assets/more_frequent_evals.png" width="80%" alt="Illustration of the value of more frequent evaluations at the start of
training.">
</p>

<p align="center"><b>Figure 5:</b> Illustration of the value of more frequent evaluations at the start of training. Useful if there’s a suspicion that the model suffers from early training instability.</p>

#### Potential fixes for common instability patterns

-   Apply learning rate warmup
    -   Best for early training instability.
-   Apply gradient clipping
    -   Good for both early and mid training instability, may fix some bad inits
        that warmup cannot.
-   Try a new optimizer
    -   Sometimes Adam can handle instabilities that Momentum can’t. This is an
        active area of research.
-   We can ensure that we’re using best practices/initializations for our model
    architecture (examples below).
    -   Add residual connections and normalization if the model doesn't contain
        it already.
-   Normalization should be the last operation before the residual. E.g. x +
    Norm(f(x)).
-   Norm(x + f(x)) known to cause issues.
-   Try initializing residual branches to 0 (e.g.
    [ReZero init](https://arxiv.org/abs/2003.04887)).
-   Lower the learning rate
    -   This is a last resort.

#### Learning rate warmup

<p align="center">
<img src="assets/instability_during_warmup.png" width="80%" alt="An example of instability during a warmup period (note the horizontal axis log
scale).">
</p>

<p align="center"><b>Figure 6:</b> An example of instability during a warmup period (note the horizontal axis log scale). 40k steps of warmup was needed for successful training in this case.</p>

##### When to apply learning rate warmup

<p align="center">
<img src="assets/axis_model_with_instability.png" width="49%" alt="Axis plot for model with instability">
</p>

<p align="center"><b>Figure 7a:</b> An example of a hyperparameter axis plot for a model exhibiting training instability. The best learning rate is at the edge of what is feasible. An "infeasible" trial is defined as one that either produces NaNs or uncharacteristically high values of the loss.</p>

<p align="center">
<img src="assets/loss_model_with_instability.png" width="49%" alt="Loss curve for model with instability">
</p>

<p align="center"><b>Figure 7b:</b> The training loss of a model trained with a learning rate where we see instability.</p>

-   Figure 7a shows a hyperparameter axis plot that indicates a model
    experiencing optimization instabilities, because the best learning rate is
    right at the edge of instability.
-   Figure 7b shows how this can be double-checked by examining the training
    loss of a model trained with a learning rate either 5x or 10x larger than
    this peak. If that plot shows a sudden rise in the loss after a steady
    decline (e.g. at step \~10k in the figure above), then the model likely
    suffers from optimization instability.

##### How to apply learning rate warmup

<p align="center">
<img src="assets/beneficial_effect_warmup.png" width="80%" alt="Beneficial effect of warmup on training instabilities">
</p>

<p align="center"><b>Figure 8:</b> Beneficial effect of learning rate warmup on addressing training instabilities.</p>

-   Using the section immediately above, we assume that the practitioner has
    already identified the learning rate at which the model becomes unstable.
    This is the `unstable_base_learning_rate`.
-   Warmup involves prepending a learning rate schedule that ramps up the
    learning rate from 0 to some stable `base_learning_rate`, that is at least
    one order of magnitude larger than `unstable_base_learning_rate`. The
    default would be to try a `base_learning_rate` that’s 10x
    `unstable_base_learning_rate`. Although note that it’d be possible to run
    this entire procedure again for something like 100x
    `unstable_base_learning_rate`. The specific schedule is:
    -   Ramp up from 0 to `base_learning_rate` over `warmup_steps`.
    -   Train at a constant rate for `post_warmup_steps`.
-   Our goal is to find the shortest number of `warmup_steps` that allows us to
    access peak learning rates that are much higher than
    `unstable_base_learning_rate`.
-   So for each `base_learning_rate`, we need to tune `warmup_steps` and
    `post_warmup_steps`. It’s usually fine to set `post_warmup_steps` to be
    `2*warmup_steps`.
-   Warmup can be tuned independently of an existing decay schedule.
    `warmup_steps` should be swept at a few different orders of magnitude. For
    example, an example study could try [10, 10<sup>3</sup>, 10<sup>4</sup>,
    10<sup>5</sup>]. The largest feasible point shouldn't be more than 10% of
    `max_train_steps`.
-   Once a `warmup_steps` that doesn't blow up training at `base_learning_rate`
    has been established, it should be applied to the baseline model.
    Essentially, we prepend this schedule onto the existing schedule, and use
    the optimal checkpoint selection discussed above to compare this experiment
    to the baseline. For example, if we originally had 10,000 `max_train_steps`
    and did `warmup_steps` for 1000 steps, the new training procedure should run
    for 11,000 steps total.
-   If long `warmup_steps` are required for stable training (>5% of
    `max_train_steps`), `max_train_steps` may need to be increased to account
    for this.
-   There isn't really a "typical" value across the full range of workloads.
    Some models only need 100 steps, while others (particularly transformers)
    may need 40k+.

#### Gradient clipping

<p align="center">
<img src="assets/gradient_clipping.png" width="80%" alt="Gradient clipping on early training instabilities">
</p>

<p align="center"><b>Figure 9:</b> Illustration of gradient clipping correcting early training instability.</p>

-   Gradient clipping is most useful when large or outlier gradient issues
    occur.
-   Clipping can fix either early training instability (large gradient norm
    early), or mid training instabilities (sudden gradient spikes mid training).
-   Sometimes longer warmup periods can correct instabilities that clipping does
    not: see [this section above](#How-to-apply-learning-rate-warmup).
    -   🤖 What about clipping during warmup?
-   The ideal clip thresholds are just above the "typical" gradient norm.
-   Here’s an example of how gradient clipping could be done:
    -   If the norm of the gradient $\left | g \right |$ is greater than the
        gradient clipping threshold $\lambda$, then do ${g}'= \lambda \times \frac{g}{\left | g \right |}$ where ${g}'$ is the new gradient.
-   Log the unclipped gradient norm during training. By default, generate:
    -   A plot of gradient norm vs step
    -   A histogram of gradient norms aggregated over all steps
-   Choose a gradient clipping threshold based on the 90th percentile of
    gradient norms.
    -   The threshold will be workload dependent, but 90% is a good starting
        point. If it doesn't work, this threshold can be tuned.
    -   🤖 What about some sort of adaptive strategy?
-   If we try gradient clipping and the instability issues remain, we can try it
    harder (i.e. make the threshold smaller).
-   Extremely aggressive gradient clipping is in essence a strange way of
    reducing the learning rate. If we find ourselves using extremely aggressive
    clipping, we probably should just cut the learning rate instead.
-   We would usually consider having >50% of the updates getting clipped somehow
    as "extremely aggressive".
-   If we need to do extremely aggressive gradient clipping to deal with our
    instability issues, then we might as well reduce the learning rate.

</details>

### Why do you call the learning rate and other optimization parameters hyperparameters? They are not parameters of any prior distribution.

<details><summary><em>[Click to expand]</em></summary>
<br>

-   It is true that the term "hyperparameter" has a precise
    [meaning](https://en.wikipedia.org/wiki/Hyperparameter) in Bayesian machine
    learning and referring to the learning rate and most of the other parameters
    we tune in deep learning as "hyperparameters" is an abuse of terminology.
-   We would prefer to use the term "metaparameter" for learning rates,
    architectural parameters, and all the other things we tune in deep learning,
    since it avoids the potential for confusion that comes from misusing the
    word "hyperparameter" (confusion that is especially likely when discussing
    Bayesian optimization where the probabilistic response surface models have
    their own true hyperparameters).
-   Unfortunately, although potentially confusing, the term hyperparameter has become
    extremely common in the deep learning community.
-   Therefore, for a document, such as this one, intended for a wide audience
    that includes many people who are unlikely to be aware of this technicality,
    we made the choice to contribute to one source of confusion in the
    field in hopes of avoiding another.
-   That said, we might make a different choice when publishing a research
    paper, and we would encourage others to use "metaparameter" instead in most
    contexts.

</details>

### Why shouldn't the batch size be tuned to directly improve validation set performance?
### 为什么我们不应该通过调整Batchsize的大小来提升验证集上的性能？

<details><summary><em>[Click to expand]</em></summary>
<br>

-   Changing the batch size *without changing any other details of the training pipeline* will often affect the validation set performance.
- 在保持其他训练设置一致的情况下更改Batchsize大小通常会影响验证集上的性能。
-   However, the difference in validation set performance between two batch sizes typically goes away if the training pipeline is optimized independently for each batch size.
- 但是如果在不同的Batchsize下都单独调整优化超参数的话，这种差别可能并不会出现。
-   The hyperparameters that interact most strongly with the batch size, and therefore are most important to tune separately for each batch size, are the optimizer hyperparameters (e.g. learning rate, momentum) and the regularization hyperparameters.
- 与Batchsize大小相互作用最强烈的超参数是优化器超参数(例如学习率，动量)和正则化超参数，因此对于每个批处理大小分别进行调优是最重要的。
    - Smaller batch sizes introduce more noise into the training algorithm due to sample variance, and this noise can have a regularizing effect. Thus, larger batch sizes can be more prone to overfitting and may require stronger regularization and/or additional regularization techniques.
    - 较小的Batchsize可以在训练过程中引入更多的噪声，这种噪声有一定的正则化效果。因此，较大的Batchsize大小可能更容易过度拟合，并且可能需要更强大的正则化和/或其他正则化技术。
- In addition, [the number of training steps may need to be adjusted](#choosing-the-batch-size-to-minimize-training-time) when changing the batch size.
- 除此之外，调整了Batchsize之后训练的步数也需要进行调整。[the number of training steps may need to be adjusted](#choosing-the-batch-size-to-minimize-training-time)
-   Once all these effects are taken into account, there is currently no convincing evidence that the batch size affects the maximum achievable validation performance (see [Shallue et al. 2018](https://arxiv.org/abs/1811.03600)).
- 当以上所有的因素都被考虑到之后，目前为止并没有显著的证据能够证明Batchsize的大小能够影响验证集上的最优性能。(see [Shallue et al. 2018](https://arxiv.org/abs/1811.03600)).

</details>

### What are the update rules for all the popular optimization algorithms?
### 常见流行的Optimizer对参数的更行步骤如下?

<details><summary><em>[Click to expand]</em></summary>

<br>

#### Stochastic gradient descent (SGD)

$$\theta_{t+1} = \theta_{t} - \eta_t \nabla \mathcal{l}(\theta_t)$$

#### Momentum

$$v_0 = 0$$

$$v_{t+1} = \gamma v_{t} + \nabla \mathcal{l}(\theta_t)$$

$$\theta_{t+1} = \theta_{t} - \eta_t v_{t+1}$$

#### Nesterov

$$v_0 = 0$$

$$v_{t+1} = \gamma v_{t} + \nabla \mathcal{l}(\theta_t)$$

$$\theta_{t+1} = \theta_{t} - \eta_t( \gamma v_{t+1} + \nabla \mathcal{l}(\theta_{t})$$

#### RMSProp

$$v_0 = 1 \text{,} m_0 = 0$$

$$v_{t+1} = \rho v_{t} + (1 - \rho) \nabla \mathcal{l}(\theta_t)^2$$

$$m_{t+1} = \gamma m_{t} + \frac{\eta_t}{\sqrt{v_{t+1} + \epsilon}}\nabla \mathcal{l}(\theta_t)$$

$$\theta_{t+1} = \theta_{t} - m_{t+1}$$

#### ADAM

$$m_0 = 0 \text{,} v_0 = 0$$

$$m_{t+1} = \beta_1 m_{t} + (1 - \beta_1) \nabla \mathcal{l} (\theta_t)$$

$$v_{t+1} = \beta_2 v_{t} + (1 - \beta_2) \nabla \mathcal{l}(\theta_t)^2$$

$$b_{t+1} = \frac{\sqrt{1 - \beta_2^{t+1}}}{1 - \beta_1^{t+1}}$$

$$\theta_{t+1} = \theta_{t} - \alpha_t \frac{m_{t+1}}{\sqrt{v_{t+1}} + \epsilon} b_{t+1}$$

#### NADAM

$$m_0 = 0 \text{,} v_0 = 0$$

$$m_{t+1} = \beta_1 m_{t} + (1 - \beta_1) \nabla \mathcal{l} (\theta_t)$$

$$v_{t+1} = \beta_2 v_{t} + (1 - \beta_2) \nabla \mathcal{l} (\theta_t)^2$$

$$b_{t+1} = \frac{\sqrt{1 - \beta_2^{t+1}}}{1 - \beta_1^{t+1}}$$

$$\theta_{t+1} = \theta_{t} - \alpha_t \frac{\beta_1 m_{t+1} + (1 - \beta_1) \nabla \mathcal{l} (\theta_t)}{\sqrt{v_{t+1}} + \epsilon} b_{t+1}$$

</details>

## Acknowledgments

-   We owe a debt of gratitude to Max Bileschi, Roy Frostig, Zelda Mariet, Stan
    Bileschi, Mohammad Norouzi, Chris DuBois and Charles Sutton for reading the
    manuscript and providing valuable feedback.
-   We reused some experimental data for several plots that were originally
    produced by Naman Agarwal for other joint research.
-   We would like to thank Will Chen for invaluable advice on the presentation of the document.
-   We would also like to thank Rohan Anil for useful discussions.

## Citing

```
@misc{tuningplaybookgithub,
  author = {Varun Godbole and George E. Dahl and Justin Gilmer and Christopher J. Shallue and Zachary Nado},
  title = {Deep Learning Tuning Playbook},
  url = {http://github.com/google/tuning_playbook},
  year = {2023},
  note = {Version 1.0}
}
```

## Contributing

-   This is not an officially supported Google product.

-   We'd love to hear your feedback!

    -   If you like the playbook, please [leave a star](https://docs.github.com/en/get-started/exploring-projects-on-github/saving-repositories-with-stars#starring-a-repository)! Or email
        deep-learning-tuning-playbook \[at\] googlegroups.com. Testimonials help
        us justify creating more resources like this.
    -   If anything seems incorrect, please file an issue to start a discussion.
        For questions or other messages where an issue isn't appropriate, please
        open a new discussion topic on GitHub.

-   As discussed in the preamble, this is a living document. We anticipate
    making periodic improvements, both small and large. If you’d like to be
    notified, please watch our repository (see [instructions](https://docs.github.com/en/account-and-profile/managing-subscriptions-and-notifications-on-github/setting-up-notifications/configuring-notifications#configuring-your-watch-settings-for-an-individual-repository)).

-   Please don't file a pull request without first coordinating with the authors
    via the issue tracking system.

### Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement (CLA). You (or your employer) retain the copyright to your
contribution; this simply gives us permission to use and redistribute your
contributions as part of the project. Head over to
<https://cla.developers.google.com/> to see your current agreements on file or
to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

### Code Reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

### Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).
