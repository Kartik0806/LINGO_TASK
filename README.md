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


## Optimization Techniques

Quantization:

The 11B Llama Vision-Language Model was quantized to 4-bit, reducing memory requirements so it fits within 12GB HBM.

Trade-off: Slight accuracy loss compared to full-precision models.

Pipeline Optimization (Reasoning Replacement):

Instead of relying on the LLM’s own reasoning for tool selection, explicit branching and prompt engineering were used.

Trade-off: Increases efficiency and reduces token usage, but sacrifices flexibility in handling unexpected cases.
