# Why Semantic Title-Based Chunking

## What it does

1. **Titles define the chunks.** Text is grouped under its headings. A chunk never crosses a heading.
2. **Each chunk carries its heading trail**, for example
   `Eye Health Guide > Glaucoma > Who is at risk for Glaucoma?`. The trail is embedded and BM25-indexed
   together with the text.
3. **Long sections are split by meaning.** When a section is over the size budget (350 tokens), it is cut
   between the two neighbouring sentences that are least similar (Qwen3 embeddings), preferring paragraph
   breaks, and this repeats until every piece fits.
4. **Tiny sections are merged** with their neighbours under the same parent heading, so "Complications:
   None." does not become a chunk on its own.

## Why it suits healthcare documents

- **Healthcare documents are written in sections.** Patient guides use "Symptoms", "Causes",
  "Treatment". Clinical notes use "History", "Hospital course", "Discharge medications". Research
  abstracts use "Methods", "Results", "Conclusions". Drug labels use "Dosage", "Warnings". A question
  usually maps to one of these sections.
- **The same sentence means different things in different sections.** "Take 500 mg twice daily" is only
  useful if you know which drug and which condition it belongs to. The heading trail keeps that with
  the chunk.
- **Mixing sections is risky.** A chunk that holds the end of one drug's dosage and the start of the
  next drug's side effects invites the LLM to combine them into a wrong answer. Section boundaries
  prevent that.
- **Body text often leaves out the topic.** Inside a glaucoma section the text says "the disease" or
  "it". Without the title, neither BM25 nor the embedding knows the passage is about glaucoma.
- **Sources are easier to check.** A citation such as "Lung Health guide > COPD > How to diagnose COPD?,
  page 9" can be verified by a reader in seconds.
- **Tables work the same way.** In a CSV each row is a record, and its name column (disease, drug) is
  its title.

## Where the other methods fall short

| Method | How it cuts | Problem with healthcare documents |
|---|---|---|
| Fixed-size | every N tokens | Cuts in the middle of sentences, lists and dose instructions. One chunk often holds parts of two or three topics. The chunk has no idea which disease or drug it is about. |
| Recursive | on paragraph, then line, then sentence breaks, packed up to N tokens | Sentences stay whole, but paragraphs from different sections are packed together, headings are separated from their text, and the topic label is still lost. |
| Semantic title-based | on headings, then on topic shifts | Depends on headings being detected. For a document with no headings at all it falls back to topic-shift splitting only. |

## Measured on the sample documents

`python -m scripts.compare_chunking` chunks the 15 sample documents three ways with the same size
budget and asks the 59 answerable test questions using **BM25 only** (no embedding model, no reranker),
so the only thing that changes is the chunking.

| Strategy | Chunks | Avg tokens | Chunks starting mid-sentence | BM25 hit@1 | BM25 hit@5 | MRR@5 | Answer density |
|---|---|---|---|---|---|---|---|
| Fixed-size | 661 | 344 | 83.4% | 81.4% | 94.9% | 0.879 | 38.3% |
| Recursive | 745 | 267 | 43.5% | 66.1% | 89.8% | 0.763 | 36.7% |
| Semantic title-based | 1272 | 170 | 0.0% | 74.6% | **96.6%** | 0.851 | **71.3%** |

- *hit@k*: the right passage is among the top k results. *Answer density*: the share of the retrieved
  chunk that is actually the answer.

What the numbers say:

- **Title-based chunks are much cleaner.** About 71% of a retrieved chunk is the answer, against about
  38% for the other two. Less noise goes to the reranker and the LLM, which is what RAGAS context
  precision and faithfulness reward.
- **Title-based chunking finds the answer most often in the top 5** (96.6%), and no chunk starts in the
  middle of a sentence.
- **Fixed-size is ahead at rank 1 on BM25 alone** (81.4% against 74.6%). Its chunks are twice as long,
  so one chunk can cover an answer plus its neighbours, and many title-based chunks in the same topic
  share the same heading words, which BM25 cannot tell apart. This is the job of the next two stages:
  dense vectors and the reranker choose between near-identical candidates, and they only need the
  right chunk to be in the candidate list.
- **Recursive chunking is the weakest here**, because it packs unrelated paragraphs together and drops
  the headings.

Limits of this test: it is lexical only, uses 59 questions, and the sample documents have clear
headings. The full comparison is the RAGAS run with the real models
(`python -m evaluation.run_ragas`), which is the number to quote.
