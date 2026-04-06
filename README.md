# regextable-python
Added to the non_financial_org_terms dictionary - lists common, ignorable words like 'university', 'association' that should not affect matching score much

no changes to get_longest_common_substring
no changes to basicHash, corpHash

cleaning now lowercases in addition to stripping metadata
added abbrev_map in clean_fin_org_names that bridges abbreviation language between compustat and cik ex: hldg->holding, co->company

no changes to get_data_row
no changes to clean_financial_measure
no changes to get_covariate_df
no changes to clean_match_score
no changes to get_quantile_by_variable
no changes to get_match_candidate_score

damerau levenshtein matching utilized in get_match_candidate_score
 - more type friendly since transpositions arent penalized as heavily
 - token level flexibility - more lenient on different orderings of same tokens
 - also considers entire string structure in addition to individual tokens
 - inverse frequency in effect as well
 - downside: time inefficient
