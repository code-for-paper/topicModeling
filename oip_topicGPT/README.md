- The v5 and v6 files need to be modified according to the actual topic list obtained
- When v6 performs regression, it analyzes different indicators and needs to comment out other indicators.
For example, when doing linear term analysis, cross-term indicators need to be commented out; when doing quadratic term analysis, cross-term indicators need to be commented out.
The prerequisite for the corresponding indicators in the cross-term list to be effective is that the corresponding linear term indicators are not commented out.
- In util.py, you can configure the base-url and api-key of the GPT model provider
