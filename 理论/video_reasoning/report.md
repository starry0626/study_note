# 提高视觉感知
## VIDEOP2R: Video Understanding from Perception to Reasoning(2025/11/14)
[alphaxiv](https://www.alphaxiv.org/abs/2511.11113?chatId=019b83d1-99e3-7e8c-8bf6-3ba404e3a661)
### 动机
本文认为, 直接对视频推理进行GRPO微调, 以最终结果确定奖励忽略了视觉问题的两个步骤: 感知与推理.
而本文的解决方案也很简单, 推理之前增加一段描述视频内容, 将感知与推理分开.
![alt text](image.png)
### 奖励
在格式与最终答案的基础上新增了一个感知奖励, 通过一个LLM衡量感知部分的表述能否得到最终的答案.
问题+感知描述+答案 -(LLM)-> 感知奖励
### 数据
162K SFT + 162K RL. 数据量有点夸张. 