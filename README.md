5.30.21 复现时的一次错误:

在Average Filling 中对于用户偏差和物品偏差的参数初始化中减去的是物品平均评分和用户平均评分, 但是在后面的算法对于用户偏差和物品偏差的初始化中都是减去平均评分, 一开始没有注意到这个细节, 在以为一致的情况下套用了AF对模型参数的初始化代码。 对结果可能会有影响。

已修正。

(但是在一些算法中的结果更好...)


6.1.21 MF-MPC复现结果错误, 后面再检查...

6.3.21 有些默认的原则: 推荐排序的时候不应该考虑原先已经存在测试集的(u, i)... 在BPR的实现里面忽略了这点导致算出来的Pre@k和Rec@k结果不理想, 已修正。

6.5.21 FISM_rmse: pointwise, FIMSE_auc: pairwise

6.6.21 复现MF-MPC时, 错误理解公式(8)和(15)中|I_r_u \ {i}|为求值的合的绝对值, 应该是取集合长度。

6.9.21 还有不对的算法: MF-MPC和FISM_auc

6.11.21 MF-MPC中结果不对是因为理解错了, 其中的M_i矩阵有r个而不是一个, 已修正, 应该能得到正确结果。

6.13.21 MF-MPC结果正确。 FISM_auc中b_j和b_i共享一个参数而且超参数设置不一致, 已修正, 希望能得到正确结果。

6.14.21 FISM_auc设置T=500时结果已经比课件好, 应该没有错误了。