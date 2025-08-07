# topicGPT Process
1. Generate raw topics
2. Merge raw topics (can use GPT, or sentence embedding and other technologies)
3. Select keywords for each topic
4. Count the number of keyword occurrences for each topic in the article, sum them up and divide by the total number of words as the score for the corresponding topic
5. Concatenate score, annual report date, financial report date lagged by one year, key financial indicators
6. Use OLS regression analysis to analyze the impact of each topic's score on the financial report

- The oip_topicGPT directory contains the main code
- oip_prompt stores the prompts needed for v1 and v3