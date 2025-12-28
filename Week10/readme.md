Week 10: Prompt Optimization & Creativity Tests 

This folder contains the experiments conducted for Week 10, focusing on how creativity parameters influence Generative AI outputs.

1. Objective
The goal of this assignment was to:

Compare different prompts and record the results.

Experiment with creativity parameters such as temperature, max_length, and top_p.

Analyze the balance between consistency and randomness in generated text.

2. Parameter Experiment Results
Based on the tests performed in the Colab notebook, here is a summary of the findings as seen in the generated outputs:
<img width="1536" height="467" alt="Ekran görüntüsü 2025-12-28 182734" src="https://github.com/user-attachments/assets/142e8880-c6ba-44d1-83f5-39c358207b62" />

3. Analysis of Key Parameters
Temperature: Higher values (1.2) significantly increased
the randomness and "creativity" of the responses, while lower values (0.1)
made the model repetitive and deterministic.

Top_p (Nucleus Sampling): Setting this to 1.0 allowed for a
wider range of vocabulary, whereas lower values restricted
the model to the most likely word choices.

Max Length: Controlled the output size to prevent the model 
from cutting off mid-sentence or rambling.

4. Conclusion
For artistic tasks like poetry, higher temperature settings are
preferable for uniqueness. For educational or factual tasks
(like explaining physics), lower to medium temperature settings
ensure the model stays on topic and avoids repetition
