# regextable-python


Get longest common substring, basichash, corpHash hasn't been modified
Added is_matchworthy_org to account entries that we should ignore "none, 01"

clean_fin_org_names:
-added abbrev_map to bridge terminologies of compustat and cik (this change was meant to try to resolve 11, so I'm unsure about this)
- Also added non financial org terms to clean

no changes to get_data_row, clean_financial_measures, get_covariate_dfs, clean_match_score, get_quantile_by_variable

get_match_candidate score:
Damerau levenshtein - distance formula is more typo-friendly, allowing for substitutions/transpositions in a token to not impact score muc
 - token scores still based off inverse frequency
 - multiplicative penalty allows for both individual token comparison and overall string comparison to be taken into account (final score = match_score * (1 - normalized-dl)) where match_score focuses on token and normalized_dl focuses on overall string structure
 - I attempt to account for downsides of the flexiblility of the distance formula by incorporating is_matchworthy_org for short names

Currently excluding nonprofits from sources

importing sources:
 - replaced if else chain that specifies column names with col_map
 - fillna(0), to_numeric, astype(int) all try to keep ID's clean (no floats or empties)
 - unique_id is created (financial_dataset_name + cleanid) to effectively be able to track back to original source
 - efficiency is optimized with org_name_df, which is giant single data frame table that allows clean_fin_org_names to be applied in one vectorized step (uses lots more RAM though)

Cascade explained:
This script utilizes multiple matching algorithms in succession to efficiently match all entries, beginning with cheapest and quickest algorithms, pulling their matches out, and matching the leftovers with more nuanced, time-consuming algorithms.

Setting up: 
- token frequency dictionary is built using Counter (way faster than loops)
- candidate match dictionary:an index that is built where each token is assigned a list of id's and names to help speed up matching (unfortunately uses lots of RAM) (GoldmanSachs wont match since this relies on perfect token splitting)
- very common words that are junk should be added to STOPWORDS to be cleaned

Stage1:Exact identity match
- strings that are exact matches are matched first
- taken out and rest of entries are carried over to stage 2
- happens in very fast sequence using Pandas Database Join

Stage 2:First-Letter match
- Identifies rarest tokens in an entry and pulls out potential matches
- strings with matching first letter are considered
- matching score is calculated

Stage 3:Flexible first-letter match
- First letter of any important tokens in the entry are considered
- Importance of token is decided by length (len > 3) and rarity

Stage 4: Relaxed fuzzy
- Considers top 3 rarest words of entry
- Ignores stopwords
- Current threshold score of 0.70

Assembly
- Loops through entire row of info (URL, submitter, agency...) instead of just cleaned name to be able to distinguish agencies
- Rejects top match if it doesn't meet threshold
- Keeps more useful info(docket_id, comment agency, url) and keeps original entries to compare
- double checks that context of name actually looks like organization using named entity recognition

Estimating Person or Organization
- instead of trying to detect person tag with spacy, script now labels entities that matched to an official financial library with score >= 0.70 as organizations
- Downside: at risk of false negatives, since we only consider entries we know for certain (>=.70) to be organizations
