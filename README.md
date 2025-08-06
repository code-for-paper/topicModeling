# topicGPT的流程
1. 生成原始的topics
2. 合并原始topic(可以使用gpt，或者sentence embedding等技术)
3. 为每个topic选取keywords
4. 统计文章的每个topic的keywords次数，加起来除以总词数作为对应topic的分数
5. 拼接score，年报日期，滞后一年的财报日期，财报的关键指标
6. 使用OLS回归分析，分析每个topic的分数对财报的影响

- oip_topicGPT目录是主要代码
- v5和v6文件需要根据实际得到的topic list来修改
- v6在做regression的时候，对不同的指标做分析，需要注释掉其他指标。
比如做一次项分析的时候，需要注释掉交叉项指标，做二次项分析的时候，需要注释掉交叉项指标。
交叉项列表中对应指标有效的前提是，一次项对应指标没有被注释。
- util.py中可以配置gpt模型供应商的base-url和api-key
