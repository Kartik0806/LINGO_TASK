# LINGO_TASK

## MODEL SELECTION: 

I experimented with multiple models, including **Llama 3.2-1B**, **lingshu-medical-mllm/Lingshu-7B**, **medgemma-4B-it**, and **ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1**. Among these, **MedGemma-4B-it** proved to be the most suitable choice for this domain.

**Justification:**

* **Domain relevance:** MedGemma is specifically fine-tuned for medical use cases, which makes it more reliable in providing accurate responses in this sensitive domain.
* **Performance:** Unlike Lingshu, which struggled with identifying whether an image was present, MedGemma correctly interpreted multimodal inputs.
* **Efficiency:** Compared to larger models (e.g., 11B Llama VLM), MedGemma offers a good balance between model size and inference speed, making it practical while still delivering high-quality outputs.
* **Reduced hallucinations:** Smaller general-purpose models like Llama 3.2-1B often produced hallucinations, whereas MedGemma was more stable and accurate.

Hence, **MedGemma-4B-it** is chosen as the most appropriate model since it is lightweight enough for efficient use but still specialized enough to provide accurate and contextually relevant responses in the medical domain.

## Model Pipeline

Input Processing:

The system first receives the user query and classifies it.

If the query is about hospital logistics, it is labeled as HOSPITALINFO.

Domain-Specific Knowledge Integration:

For HOSPITALINFO queries, relevant details (logistics, helplines) are retrieved from a structured file.

If not logistics-related, the query is compared with the FAQ knowledge base using vector similarity.

If similarity exceeds a threshold, the matched FAQ answer is retrieved; otherwise, no external context is added.

Model Inference:

The query, along with any retrieved context, is concatenated through prompt engineering.

This combined input is then passed to the LLM for response generation.

Output Formatting:

The model produces a structured, user-friendly answer.

Retrieval Mechanism:
Both FAQ and hospital logistics rely on vector databases for efficient semantic search. A branching logic ensures the correct retriever is used, minimizing hallucinations and tool misuse.

## Prompt Design
### Initial Version (System Prompt Only)  

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
Issues:

Too broad and overloaded: one prompt tried to handle classification, FAQ usage, and fallback knowledge.

Produced inconsistent results — sometimes mixed hospital info with general answers.

System prompt reduced flexibility; small changes in query style led to poor performance.

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

Improvements:

Broke down logic into smaller, specialized prompts instead of a single overloaded system prompt.

Clearer instructions → reduced hallucinations and irrelevant answers.

Explicit fallback rules → ensured robust handling when FAQ context was missing or irrelevant.

Overall: pipeline became faster, more accurate, and easier to maintain.
## Optimization Techniques

Quantization:

The 11B Llama Vision-Language Model was quantized to 4-bit, reducing memory requirements so it fits within 12GB HBM.

Trade-off: Slight accuracy loss compared to full-precision models.

Pipeline Optimization (Reasoning Replacement):

Instead of relying on the LLM’s own reasoning for tool selection, explicit branching and prompt engineering were used.

Trade-off: Increases efficiency and reduces token usage, but sacrifices flexibility in handling unexpected cases.
