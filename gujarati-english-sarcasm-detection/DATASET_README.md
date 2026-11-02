# Dataset Documentation: Gujarati-English (Romanized) Sarcasm Detection Dataset

**File**: `sarcasm_dataset.csv`  
**Version**: 2.0.0 (Leakage-Free, Group-Contained)  
**Target Domain**: Gujarati-English code-mixed and Romanized Gujarati informal conversational text, social media commentary, complaints, and customer reviews.  
**Task**: Binary Sarcasm Classification (`Sarcasm` vs `Non-Sarcasm`).

---

## 1. Dataset Overview & Key Statistics

| Attribute | Value |
| :--- | :--- |
| **Total Samples** | **1,200** |
| **Class Balance** | Exactly **600 Sarcasm** (50.0%) / **600 Non-Sarcasm** (50.0%) |
| **Unique Semantic Groups (`group_id`)** | **241 distinct isolated groups** (117 Sarcasm / 124 Non-Sarcasm) |
| **Data Leakage Prevention** | **Group-ID separation**: Generated spelling variants inherit their assigned parent `group_id`. The split has **zero observed group-ID overlap**. Group assignments were not independently audited for every paraphrase/template relation. |
| **Language Coverage** | Gujarati-English Code-Mixed (~96.1%), Pure Romanized Gujarati (~3.9%) |
| **Deduplication** | Strict exact duplicate removal (`drop_duplicates` on sentence text) |
| **Orthographic Noise** | Group-tied phonetic spelling variations (e.g., `che`/`chhe`, `su`/`shu`, `nathi`/`nai`, `thayu`/`thyu`, `maja`/`majja`) |

---

## 2. Dataset Schema

The dataset is saved in standard UTF-8 CSV format with the following columns:

| Column | Type | Description | Predictive Feature? |
| :--- | :--- | :--- | :--- |
| `id` | Integer | Unique identifier for each record (1 to 1200) | **No** (Row identifier only) |
| `sentence` | String | Raw text input written in Latin (Roman) script | **Yes** (Primary input feature) |
| `label` | String | Ground-truth binary target label (`Sarcasm` or `Non-Sarcasm`) | **Target Variable** |
| `source_type` | String | Provenance category (`programmatically_authored`, `spelling_variant`) | **No** (Provenance metadata) |
| `language` | String | Language mix type (`code_mixed_guj_eng`, `gujarati_romanized`) | **No** (Provenance metadata) |
| `sub_category` | String | Linguistic typology of the sentence (e.g., `mock_praise`, `sentiment_contrast`, `genuine_positive`) | **No** (Diagnostic metadata) |
| `group_id` | Integer | Semantic cluster ID linking related sentences/variants to prevent leakage | **No** (Split-grouping key only) |

---

## 3. Class Distribution & Typology

To avoid superficial keyword memorization (e.g., assuming any sentence with positive words like *"mast"* or *"superb"* is sarcastic, or any sentence with negative words like *"fail"* or *"traffic"* is non-sarcastic), the dataset is structured with balanced, complementary sub-categories:

### Sarcasm Sub-categories (600 Samples across 117 Groups)
1. **Mock Praise (479 samples)**:
   - Literal wording uses high-praise vocabulary (*"wah"*, *"kya baat che"*, *"superb"*, *"mast"*, *"bullet train speed"*, *"salute"*) juxtaposed with terrible service, excessive delays, or defective products.
   - *Example*: `"Wah, su mast service che, 3 kalak pachi thandi chai aapi!"`
   - *Example*: `"Super fast delivery, 10 divas ma parcel aavyu, ekdum zabardast."`
2. **Sentiment Contrast & Irony (57 samples)**:
   - Acute frustration, disaster, or failure followed immediately by an ironic positive evaluation.
   - *Example*: `"Exam ma fail thai ne maja aavi."`
   - *Example*: `"Aaje traffic ma be kalak waste thaya, ekdum perfect."`
   - *Example*: `"Garmi 46 degree che ane light chali gai, full AC feel aave che."`
3. **Rhetorical Ridicule & Feigned Politeness (62 samples)**:
   - Mocking questions, inflated ego teasing, or feigned modesty.
   - *Example*: `"Tari vaat sambhali ne toh duniya badlai gai, Nobel prize aapo aane koi."`
   - *Example*: `"Tame toh mota celebrity thai gaya have, amari jeva aam mitra ne kya thi yaad karo."`
4. **Understatement Irony (2 samples)**:
   - Extreme understatements minimizing severe disruption.
   - *Example*: `"Flight canceled, luggage lost, and no hotel provided, minor inconvenience by airline!"`

### Non-Sarcasm Sub-categories (600 Samples across 124 Groups)
1. **Factual & Neutral Informational (269 samples)**:
   - Straightforward statements of schedule, banking hours, transit routes, municipal procedures, or technical documentation.
   - *Example*: `"Aaje somvar che ane office javu padshe."`
   - *Example*: `"Flight nu schedule change thayu che, sanje 6 vagye departure che."`
   - *Example*: `"Ahmedabad metro Thaltej to Vastral route par daily services morning 6:30 AM thi operational che."`
2. **Genuine Positive & Praise (178 samples)**:
   - Direct, earnest praise, gratitude, or achievement without ironic intent, containing positive words used literally.
   - *Example*: `"Wah, kharekhar khub saras kaam karyu che, aabhar!"`
   - *Example*: `"Wah, su mast ras-puri banavya che aaje mom, ekdum maja aavi gai!"`
   - *Example*: `"Finally kaam puru thayu, what a relief!"`
3. **Genuine Frustration & Direct Complaints (153 samples)**:
   - True negative sentiment expressing anger, disappointment, pain, or stress without sarcasm or irony.
   - *Example*: `"Aaje traffic ma be kalak waste thaya, hu ekdum thaki gayo chu."`
   - *Example*: `"Exam ma fail thai gayo, have ghar par su javab aapis?"`
   - *Example*: `"Laptop motherboard short circuit thai gayu ane badho data lost thai gayo, bohot dukh che."`

---

## 4. Provenance and Annotation Limitations

1. **Programmatically authored conversational-style data (802 samples)**:
   - Programmatically authored to resemble conversational styles. These are synthetic examples, not documented samples collected from real users.
   - No independent native-speaker annotation evidence is included in this project. Labels should be treated as provisional.
2. **Group-Tied Spelling Variations (398 samples)**:
   - Romanized Gujarati lacks standardized spelling. To ensure the model learns robust character-level representations, common phonetic alternations were generated:
     - `che` $\leftrightarrow$ `chhe`
     - `su` $\leftrightarrow$ `shu`
     - `nathi` $\leftrightarrow$ `nai`
     - `thayu` $\leftrightarrow$ `thyu`
     - `maja` $\leftrightarrow$ `majja`
   - **Crucial Integrity Rule**: Every spelling variant is assigned the **same `group_id`** as its parent sentence. This keeps each generated spelling variant with its assigned parent across the split.
3. **Leakage Prevention**:
   - `GroupShuffleSplit` partitions the 241 groups cleanly.
   - Zero exact duplicates exist (`drop_duplicates` applied).
   - Training, validation, and test splits have 0% group overlap.

---

## 5. Limitations & Ethical Considerations

- **Colloquial Evolution**: Informal Romanized Gujarati evolves rapidly; new slang terms and internet abbreviations may require continuous dataset expansion.
- **Context Dependency**: Sarcasm in single-sentence snippets can occasionally rely on shared unstated real-world context (e.g., knowing whether an exam was easy or difficult).
- **Phonetic Ambiguity**: Transliteration ambiguities exist (e.g. `kaam` vs `kam`), requiring character-level n-gram features to disambiguate.
