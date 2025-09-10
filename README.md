### Model Selection

Some other models: **AdaptLLM/biomed-Qwen2-VL-2B-Instruct**, **Qwen/Qwen2-VL-2B-Instruct**

I experimented with multiple models for this project, including **Llama 3.2-1B**, **lingshu-medical-mllm/Lingshu-7B**, **medgemma-4B-it**, and **ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1**. **MedGemma-4B-it** proved to be the most suitable choice only if moder GPUs. For older ones either use half precision **Qwen/Qwen2-VL-2B-Instruct** or 4-bit quantized **Llama-3.2-11B-Vision-Instruct** which takes upto 12GB of High Bandwith Memory in GPU.

**Justification:**

  * **Domain Relevance:** MedGemma is fine-tuned for medical use cases, making it more reliable and accurate in this sensitive domain.
  * **Performance:** Unlike Lingshu, which struggled to identify images, MedGemma correctly interpreted multimodal inputs.
  * **Efficiency:** Compared to larger models, MedGemma offers a good balance between size and speed.
  * **Reduced Hallucinations:** Smaller general-purpose models often produced unreliable information, while MedGemma was more stable and accurate.

-----

### Model Pipeline

The system first receives the user's query and classifies it.

  * **Input Processing:** If the query is about hospital logistics, it is labeled as `HOSPITALINFO`.
  * **Domain-Specific Knowledge Integration:** For `HOSPITALINFO` queries, relevant details are retrieved from a structured file. If not logistics-related, the query is compared with an FAQ database using vector similarity. If a high similarity is found, the matched FAQ answer is retrieved.
  * **Model Inference:** The query, along with any retrieved context, is combined through prompt engineering. This combined input is then passed to the LLM for response generation.
  * **Output Formatting:** The model produces a structured, user-friendly answer.
  * **Retrieval Mechanism:** Both FAQ and hospital logistics systems rely on **vector databases** for efficient semantic search. A branching logic ensures the correct retriever is used, which minimizes hallucinations and tool misuse.

-----

### Prompt Design

I initially used a single, broad system prompt, but it was too overloaded and produced inconsistent results. The solution was to create smaller, specialized prompts.

**Initial Version (System Prompt Only):**

```python
system_prompt = """
You are a helpful medical assistant.
Answer {Yes}, if the Query is about Offering hospital/clinic support info
(e.g., working hours, appointment booking info, contact numbers).

Else If: context is provided from FAQ database.
You provide the answer from the context provided between <ctx> </ctx>.

Else:
If no context is provided, you can use your knowledge to answer.
"""
```

**Issues:**

  * Too broad and overloaded; it tried to handle classification, FAQ usage, and fallback knowledge all at once.
  * Produced inconsistent results, sometimes mixing hospital info with general answers.
  * The system prompt reduced flexibility; minor changes in the query led to poor performance.

**Improved Version (Specialized Prompts):**

Breaking down the logic into smaller prompts provided a better solution.

```python
classification_prompt = """
Classify the user query strictly into one category:
- HOSPITAL_INFO → hospital/clinic logistics (hours, appointments, contacts, directions, doctors available).
- GENERAL → all other queries.

Return only the category name.
User query: {query}
Category:
"""

help_prompt = """
Look at the Context and Query, refer to that context and answer relevant part only.
If the Context and Query are totally unrelated, say "I don't have answer to this question".

Context: {context}
Query: {query}
"""

faq_prompt = """
Look at the Context and Query, if Context contains answer to the query, refer to that context only.
Otherwise use your knowledge to answer the query.

Context: {context}
Query: {query}
"""
```

**Improvements:**

  * Clearer instructions reduced hallucinations and irrelevant answers.
  * Explicit fallback rules ensured robust handling when FAQ context was missing.
  * The pipeline became faster, more accurate, and easier to maintain.

-----

### Optimization Techniques

I used a couple of techniques to optimize the pipeline's performance.

  * **Quantization:** The 11B Llama Vision-Language Model was quantized to 4-bit. This reduced memory requirements, allowing it to fit within 12GB of HBM, though it resulted in a slight loss of accuracy.
  * **Pipeline Optimization (Reasoning Replacement):** Instead of relying on the LLM’s own reasoning for tool selection, I used explicit branching and prompt engineering. This increases efficiency and reduces token usage but sacrifices some flexibility in handling unexpected cases.
