# Multimodal Instructional Assistant — Process Diagrams

---

## 1. System Architecture

```mermaid
graph TD
    UI["🖥️ Gradio Web UI\napp.py — localhost:7860"]
    IT["🖼️ Interactive Tab\nSingle Image\nCaption / Q&A"]
    BT["📦 Batch Tab\nMultiple Images\nCSV Export"]
    FH["Florence2Handler\nmodel_handler.py"]
    F2["Florence-2-large\n🟢 GPU CUDA\n~0.83 GB VRAM\nOCR + Captioning"]
    TL["TinyLlama-1.1B\n🟢 GPU CUDA\n~2.2 GB VRAM\nText Reasoning"]

    UI --> IT
    UI --> BT
    IT --> FH
    BT --> FH
    FH --> F2
    FH --> TL

    style UI fill:#1e3a5f,color:#fff
    style IT fill:#2d5986,color:#fff
    style BT fill:#2d5986,color:#fff
    style FH fill:#1a4731,color:#fff
    style F2 fill:#145a32,color:#fff
    style TL fill:#145a32,color:#fff
```

---

## 2. App Startup Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant A as app.py
    participant FH as Florence2Handler
    participant F2 as Florence-2 (GPU)
    participant TL as TinyLlama (GPU)
    participant UI as Gradio UI

    U->>A: python app.py
    A->>FH: Florence2Handler()
    Note over FH: TechnicalCaptionEnhancer() created (empty)
    A->>FH: handler.load_model()
    FH->>F2: AutoProcessor.from_pretrained()
    F2-->>FH: Processor ready
    FH->>F2: AutoModelForCausalLM → float16 → .to(cuda)
    F2-->>FH: Florence-2 loaded ✅ (~20s)
    A->>TL: handler.enhancer.load()
    TL-->>A: TinyLlama ready ✅ (~10s)
    A->>UI: demo.launch(port=7860)
    UI-->>U: http://localhost:7860
```

---

## 3. Caption Pipeline

```mermaid
flowchart TD
    A([User uploads image\n+ clicks Generate]) --> B[process_single\napp.py]
    B --> C[handler.caption\nmodel_handler.py]

    C --> D[Florence-2 OCR\n&lt;OCR&gt; task token\n~1s on GPU]
    C --> E[Florence-2 Visual Caption\n&lt;MORE_DETAILED_CAPTION&gt;\n~2s on GPU]

    D --> F{OCR text\n≥ 3 words?}
    E --> F

    F -- NO --> G([Return Florence-2\nvisual caption\n⏱ ~3s total])

    F -- YES --> H[TinyLlama on GPU\nPrompt: labels + visual desc\nExplain in 2-3 sentences]
    H --> I([Return enhanced\ntechnical explanation\n⏱ ~16s total])

    style A fill:#2d5986,color:#fff
    style G fill:#145a32,color:#fff
    style I fill:#145a32,color:#fff
    style H fill:#1a4731,color:#fff
    style D fill:#4a235a,color:#fff
    style E fill:#4a235a,color:#fff
```

---

## 4. Q&A Generation Pipeline

```mermaid
flowchart TD
    A([User clicks Generate\nQ&A mode]) --> B[handler.generate_questions\nmodel_handler.py]

    B --> C[Florence-2 OCR\n&lt;OCR&gt;\n~1s GPU]
    B --> D[Florence-2 Caption\n&lt;MORE_DETAILED_CAPTION&gt;\n~2s GPU]

    C --> E{OCR text\n≥ 3 words?}
    D --> E

    E -- NO --> F([5 generic fallback\nquestions returned\n⏱ ~3s total])

    E -- YES --> G[TinyLlama on GPU\nWrite 5 numbered\nstudy questions\nfor this diagram]

    G --> H([5 specific educational\nquestions returned\n⏱ ~25s total])

    style A fill:#2d5986,color:#fff
    style F fill:#145a32,color:#fff
    style H fill:#145a32,color:#fff
    style G fill:#1a4731,color:#fff
    style C fill:#4a235a,color:#fff
    style D fill:#4a235a,color:#fff
```

---

## 5. Batch Processing Pipeline

```mermaid
flowchart TD
    A([User uploads N images\n+ clicks Process All]) --> B[process_batch\napp.py]
    B --> C[Load all images\nfrom disk]
    C --> D{For each image\nsequentially}

    D --> E{Mode?}
    E -- Caption --> F[handler.caption\nOCR + TinyLlama pipeline]
    E -- Q&A --> G[handler.generate_questions\nOCR + TinyLlama pipeline]

    F --> H[Store result\nin DataFrame row]
    G --> H
    H --> I{More\nimages?}
    I -- Yes --> D
    I -- No --> J[Build DataFrame]
    J --> K[Save to\noutput/batch_results.csv]
    K --> L([Show table in UI\n+ Download CSV button])

    style A fill:#2d5986,color:#fff
    style L fill:#145a32,color:#fff
    style F fill:#1a4731,color:#fff
    style G fill:#1a4731,color:#fff
```

---

## 6. VRAM Memory Layout

```mermaid
pie title GPU VRAM Usage (4.29 GB total)
    "Florence-2-large (0.83 GB)" : 0.83
    "TinyLlama-1.1B (2.2 GB)" : 2.2
    "Free headroom (1.26 GB)" : 1.26
```
