# regextable-python


Get longest common substring, basichash, corpHash hasn't been modified
Added is_matchworthy_org to account entries that we should ignore "none, 01"

clean_fin_org_names:
-added abbrev_map to bridge terminologies of compustat and cik
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

Stage1:Exact identity match

Stage 2:First-Letter match

Stage 3:Flexible first-letter match

Stage 4: Relaxed fuzzy(original algorithm)
