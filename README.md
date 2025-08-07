# topicGPT的流程
1. 生成原始的topics
2. 合并原始topic(可以使用gpt，或者sentence embedding等技术)
3. 为每个topic选取keywords
4. 统计文章的每个topic的keywords次数，加起来除以总词数作为对应topic的分数
5. 拼接score，年报日期，滞后一年的财报日期，财报的关键指标
6. 使用OLS回归分析，分析每个topic的分数对财报的影响

- oip_topicGPT目录是主要代码
- oip_prompt存的是v1和v3需要用到的prompt