# Discarded oracle review — pattern analysis
- Source: `datasets/train_samples/discarded_oracle_review.csv`
- Rows: 2,148 (agent runs) | Questions: 358
- By dataset (runs): {'hotpot': 966, 'math': 654, 'musique': 528}
## Aggregate heuristics (per question, best over 6 agents)
| Signal | Questions | % of discarded |
|---|---:|---:|
| q_total | 358 | 100.0% |
| high_token_overlap_ge_0.85 | 19 | 5.3% |
| high_seq_ratio_ge_0.92 | 31 | 8.7% |
| substring_containment | 100 | 27.9% |
| prefix_nickname_like | 5 | 1.4% |
| medium_overlap_0.60_0.85 | 27 | 7.5% |
| any_prose_dump | 75 | 20.9% |
| all_unknown_or_empty | 0 | 0.0% |

## By dataset (question counts)
- **math**: 109 discarded questions
- **hotpot**: 161 discarded questions
- **musique**: 88 discarded questions

## Likely *semantically* acceptable (needs human / benchmark policy)
These are **not** marked correct by the current grader but match common “good enough” notions.

### 1. Prefix / shortened name vs full gold (nickname / legal name)
Count: **5** questions where some agent’s tokens are a prefix of gold tokens (≥2 tokens), e.g. “Brann Dailor” vs “Brann Timothy Dailor”.

- `train_0576` — gold: `film director and producer` … pred: `film director` (debate)
- `train_0586` — gold: `Johnny Knoxville and Spike Jonze` … pred: `Johnny Knoxville` (react)
- `train_0702` — gold: `Cumberland County, Maine, United States` … pred: `Cumberland County` (self_consistency)
- `train_0912` — gold: `Las Vegas Boulevard South` … pred: `Las Vegas Boulevard` (self_consistency)
- `train_0955` — gold: `YG Entertainment Inc.` … pred: `YG Entertainment` (self_consistency)

### 2. High token overlap but not exact NEM
Jaccard(token sets) ≥ 0.85 on at least one agent: **19** questions.

- `train_0943` [hotpot] — gold `film producer and director` vs best pred fragment `director`
- `train_0015` [math] — gold `\pi^2` vs best pred fragment `To find the area bounded by the graph of \( y = \arccos(\sin x) \) and`
- `train_0029` [math] — gold `\sqrt{10}` vs best pred fragment `sqrt(10)`
- `train_0049` [math] — gold `1,722` vs best pred fragment `1722`
- `train_0071` [math] — gold `28\sqrt{3}` vs best pred fragment `28*sqrt(3)`
- `train_0075` [math] — gold `100\pi` vs best pred fragment `100*pi`
- `train_0099` [math] — gold `35,280` vs best pred fragment `35280`
- `train_0129` [math] — gold `12,441,600` vs best pred fragment `12441600`

### 3. Substring containment (after squash)
**100** questions where pred⊂gold or gold⊂pred (length ≥ 3).

- `train_0515` — gold `biophysicist` … pred `physiologist and biophysicist` (react)
- `train_0516` — gold `Travis Lazarus "Travie" McCoy` … pred `Travie McCoy` (self_consistency)
- `train_0527` — gold `classified information from the National Security Agency` … pred `Classified information from the National Security Agency was leaked first in Jun` (react)
- `train_0533` — gold `Joshua Bishop "Josh" Kelley Sr.` … pred `Josh Kelley` (self_consistency)
- `train_0537` — gold `Sydney based media production company Kapow Pictures` … pred `a production company` (self_consistency)
- `train_0538` — gold `Carnatic Musician` … pred `Carnatic music` (self_consistency)
- `train_0539` — gold `grazier` … pred `near a grazier in South Australia` (self_consistency)
- `train_0545` — gold `Sabermetrics` … pred `sabermetric` (multiagent)
- `train_0551` — gold `United States Secretary of Agriculture` … pred `Secretary of Agriculture` (self_consistency)
- `train_0556` — gold `China` … pred `Both are districts in China; Sanshui is in Guangdong, Rushan is in Shandong.` (raw)

### 4. Unparseable / refusal / empty cluster
**0** questions where every agent returned unknown/empty-like.


### 5. Prose instead of short span (grading + prompt mismatch)
**75** questions had ≥1 agent with long CoT-style text instead of a short answer. Often *contains* the right token but fails strict extractors / EM.


### 6. Math — LaTeX gold vs scalar / partial answer
Heuristic sample: **32** math discarded questions where gold looks like LaTeX and at least one agent returned a plain number (matrix/rank confusion, boxed mismatch).


### 7. Hotpot — state vs country (geographic precision)
Heuristic (Mexican state gold + some agent answered “Mexico”): **1** discarded question — `train_0508` (gold `Tamaulipas`, preds mostly `Mexico`). Right *country*, wrong *granularity* for Hotpot EM.


### 8. Date / year near-misses
Year tokens from gold appear in some agent text for many date questions; off-by-one calendar days (e.g. Aug 8 vs Aug 9) need **date parsing**, not token Jaccard.


## Recommendations (if you want more “accepted” answers)
1. **Aliases file** per dataset (Hotpot gold often full legal name; models use stage name).
2. **SQuAD-style F1 threshold** in addition to NEM for open QA (official Hotpot reports both).
3. **Extract last short span** from prose before EM (or stricter JSON-only scoring).
4. **Math**: keep `is_equiv` + base suffix; add optional **matrix / set** equivalence only if you define it.
5. **Geo**: allow country when gold is a subdivision *and* question asked for sovereign state (policy choice).

## Math discarded — gold shape (109 questions)

| Gold pattern | Count |
|---|---:|
| latex_matrix_like_gold | 6 |
| latex_other_gold | 60 |
| numeric_gold | 38 |
| other_gold | 5 |

Questions where **any** agent emitted “to solve…”: **29**.

## Hotpot discarded — gold contains a year (161 questions)

Approx. **11** / 161 hotpot discarded gold strings contain a 4-digit year (date / era questions).

## MuSiQue discarded — short numeric-ish gold (heuristic)

**33** / 88 questions have short gold with digits (often dates or counts).
