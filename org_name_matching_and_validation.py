#!/usr/bin/env python
# coding: utf-8

import math
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import pandas as pd
from datetime import datetime
import pickle
import re
from pyxdameraulevenshtein import damerau_levenshtein_distance
import apsw
import sys
import numpy as np
import corp_simplify_utils
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadr
from collections import Counter

# nlp
import spacy
from spacy import displacy
from collections import Counter
# to install: $python3 -m spacy download en_core_web_lg
import en_core_web_lg

# analysis/regressions
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Poisson
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

# from statsmodels.graphics.gofplots import qqplot_2samples
from scipy import stats
from joypy import joyplot
from matplotlib import cm

from datetime import date
today_for_filenames = date.today()
curr_date = str(today_for_filenames.strftime("%Y%m%d"))


NUMBER_OF_MATCHES_TO_RECORD = 10
punc_remove_re = re.compile(r'\W+')
corp_re = re.compile('( (group|holding(s)?( co)?|inc(orporated)?|ltd|l ?l? ?[cp]|co(rp(oration)?|mpany)?|s[ae]|plc))+$')
and_re = re.compile(' & ')
punc1_re = re.compile(r'(?<=\S)[\'’´\.](?=\S)')
punc2_re = re.compile(r'[\s\.,:;/\'"`´‘’“”\(\)\[\]\{\}_—\-?$=!]+')

STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS.remove("am")
STOPWORDS.remove("up")
STOPWORDS.remove("in")
STOPWORDS.remove("on")
STOPWORDS.remove("all")
STOPWORDS.remove("any")
STOPWORDS.remove("most")
STOPWORDS.remove("no")
STOPWORDS.remove("nor")
STOPWORDS.remove("own")
STOPWORDS.remove("same")
STOPWORDS.remove("so")
STOPWORDS.remove("very")
STOPWORDS.remove("s")
STOPWORDS.remove("t")
STOPWORDS.remove("d")
STOPWORDS.remove("ll")
STOPWORDS.remove("m")
STOPWORDS.remove("o")
STOPWORDS.remove("re")
STOPWORDS.remove("ve")
STOPWORDS.remove("y")

#compile regex patterns to reuse
STOPWORD_RE = re.compile(r'\b(the|of|and|in|on)\b', re.IGNORECASE)
CORP_SUFFIX_RE = re.compile(r'\b(inc|corp|ltd|llc|plc|co|company|limited)\b', re.IGNORECASE)
PDF_PATTERN_RE = re.compile(r'\s[0-9]*\s[km]b\s*pdf', re.IGNORECASE)
PUNCT_RE = re.compile(r'[^\w\s-]')  # match punctuation
MULTISPACE_RE = re.compile(r'\s+')

stopword_re_str = r""
for word in STOPWORDS:
	stopword_re_str += r'\b' + word + r'\b|'
stopword_re = re.compile(stopword_re_str[:-1]) # The negative 1 is for the fencepost |

NON_FINANCIAL_ORG_TERMS = [
    'university', 'college', 'school', 'institute', 'academy', 
    'hospital', 'medical center', 'health system', 'center', 
    'commission', 'authority', 'association', 'society', 
    'foundation', 'transportation services', 'district', 
    'chamber', 'commerce', 'library', 'museum', 'public', 
    'city', 'county', 'town', 'government', 'state', 'federal',
    'ministry', 'department', 'office'
]

NON_FINANCIAL_RE = re.compile(r'\b(' + '|'.join(NON_FINANCIAL_ORG_TERMS) + r')\b', re.IGNORECASE)

BASE_DIR = "/Users/aawesomez/Documents/UROP/NLP-regextable/"
# BASE_DIR = "/Users/jameschen/Team Name Dropbox/James Chen/JLW-FINREG-PARTICIPATION/"
# BASE_DIR = "/Users/jameschen/Documents/Code/JLW-FINREG-PARTICIPATION/"
# DB_PATH = BASE_DIR + "data/master.sqlite"
DB_PATH = BASE_DIR + "Data/master.sqlite"
# LAST_SAVE_DATASET_DATE = "20210824"
LAST_SAVE_DATASET_DATE = "20220402" # Needs to be set to the last date the 'rebuild datasets' part of this code was run

# Function to calculate longest common substring, from https://www.geeksforgeeks.org/print-longest-common-substring/
# function to find and print 
# the longest common substring of
# X[0..m-1] and Y[0..n-1]
def get_longest_common_substring(X, Y, m, n):
 
    # Create a table to store lengths of
    # longest common suffixes of substrings.
    # Note that LCSuff[i][j] contains length
    # of longest common suffix of X[0..i-1] and
    # Y[0..j-1]. The first row and first
    # column entries have no logical meaning,
    # they are used only for simplicity of program
    LCSuff = [[0 for i in range(n + 1)]
                 for j in range(m + 1)]
 
    # To store length of the
    # longest common substring
    length = 0
 
    # To store the index of the cell
    # which contains the maximum value.
    # This cell's index helps in building
    # up the longest common substring
    # from right to left.
    row, col = 0, 0
 
    # Following steps build LCSuff[m+1][n+1]
    # in bottom up fashion.
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                LCSuff[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                if length < LCSuff[i][j]:
                    length = LCSuff[i][j]
                    row = i
                    col = j
            else:
                LCSuff[i][j] = 0
 
    # if true, then no common substring exists
    if length == 0:
        return ""
 
    # allocate space for the longest
    # common substring
    resultStr = ['0'] * length
 
    # traverse up diagonally form the
    # (row, col) cell until LCSuff[row][col] != 0
    while LCSuff[row][col] != 0:
        length -= 1
        resultStr[length] = X[row - 1] # or Y[col-1]
 
        # move diagonally up to previous cell
        row -= 1
        col -= 1
 
    # required longest common substring
    longest_common_substring = ''.join(resultStr)

    return longest_common_substring


# Function from Brad Hackinen's NAMA
def basicHash(s):
    '''
    A simple case and puctuation-insensitive hash
    '''
    s = s.lower()
    s = re.sub(and_re,' and ',s)
    s = re.sub(punc1_re,'',s)
    s = re.sub(punc2_re,' ',s)
    s = s.strip()

    return s

# Function from Brad Hackinen's NAMA
def corpHash(s):
    '''
    A hash function for corporate subsidiaries
    Insensitive to
        -case & punctation
        -'the' prefix
        -common corporation suffixes, including 'holding co'
    '''
    s = basicHash(s)
    if s.startswith('the '):
        s = s[4:]

    s = re.sub(corp_re,'',s,count=1)

    return s

# function to clean org names
def clean_fin_org_names(name: str) -> str:
    if name is None or not isinstance(name, str) or name == "NA":
        return ""
    
    # James strip metadata from name
    name = name.split(',')[0]
    #Remove patterns like "10 kb pdf"
    name = PDF_PATTERN_RE.sub("", name)

    #Unicode and punctuation cleanup
    name = corp_simplify_utils.normalize_unicode(name)
    name = PUNCT_RE.sub(" ", name)

    #Remove corporate suffixes and stopwords and non-financial entity
    name = CORP_SUFFIX_RE.sub("", name)
    name = NON_FINANCIAL_RE.sub("", name)
    name = STOPWORD_RE.sub("", name)

    #Normalize spacing and lowercase
    name = MULTISPACE_RE.sub(" ", name).strip().lower()

    return name


# Function for organizing the covariates available for each of the gathered datasets
def get_data_row(match_type, match_row_num, match_on_type):

    df = covariate_dfs[match_type]
    column_names = df.columns
    match_covariates = df.iloc[match_row_num]
    
    covariate_dict = {'row_id':match_row_num, 'row_type':match_type}
    for elem_idx, elem in enumerate(match_covariates):
        var_name = match_type + "-" + match_on_type + ":" + column_names[elem_idx]
        val = elem
        covariate_dict[var_name] = val

    return covariate_dict
    
# Function to clean numeric fields that may have text like K for thousand
def clean_financial_measure(x):
    if x is None or x is np.nan or pd.isnull(x) or x == "":
        return np.nan
    elif isinstance(x, str) and not x.isnumeric():
        unit_multiplier = 1
        if "B" in x:
            x = x[:-1]
            unit_multiplier = 1000000000
        if "M" in x:
            x = x[:-1]
            unit_multiplier = 1000000
        if "K" in x:
            x = x[:-1]
            unit_multiplier = 1000
        x = x.replace(",", "")
        try:
            x = float(x) * unit_multiplier
            return x
        except:
            return np.nan
    else:
        return float(x)
    
# 3.1: Read the gathered datasets in as one dataframe
def get_covariate_dfs():
    covariate_dfs = {}
    financial_datasets = [("data/merged_resources/", "FDIC_Institutions"), 
                    ("data/merged_resources/", "FFIECInstitutions"),
                    ("data/", "CreditUnions"),
                    ("data/merged_resources/", "compustat_resources"),
                    ("data/merged_resources/", "nonprofits_resources"),
                    ("data/merged_resources/", "SEC_Institutions")
    ]
    for financial_dataset_tuple in financial_datasets:
        df = pd.read_csv(BASE_DIR + financial_dataset_tuple[0] + financial_dataset_tuple[1] + ".csv")
        covariate_dfs[financial_dataset_tuple[1]] = df
        
    # Read in opensecrets dataseparately to deal with quotechar
    df = pd.read_csv(BASE_DIR + "data/merged_resources/opensecrets_resources_jwVersion.csv", quotechar='"')
    covariate_dfs['opensecrets_resources_jwVersion'] = df
        
    # Merge compustat data to cik data on cik
    cik_df = pd.read_csv(BASE_DIR + "data/merged_resources/CIK.csv", dtype={"CIK":str})
    compustat_df = pd.read_csv(BASE_DIR + "data/merged_resources/compustat_resources.csv", dtype={"cik":str})
    compustat_df.sort_values(by=['year2', 'year1'], ascending=True, inplace=True)
    compustat_df = compustat_df.drop_duplicates(subset='cik', keep='last', ignore_index=True)
    compustat_df = compustat_df[['cik', 'marketcap']]

    # James: dtype convert
    cik_df['cik']= cik_df['cik'].astype('Int64')
    compustat_df['cik']= compustat_df['cik'].astype('Int64')

    cik_merged_df = cik_df.merge(compustat_df, how='left', left_on='cik', right_on='cik')
    del cik_merged_df['cik']
    covariate_dfs['CIK'] = cik_merged_df

    return covariate_dfs
    

def clean_match_score(x):
    if x is None or x is np.nan or pd.isnull(x) or x == "":
        return np.nan
    elif isinstance(x, str) and not x.isnumeric():
        unit_multiplier = 1
        if "B" in x:
            x = x[:-1]
            unit_multiplier = 1000000000
        if "M" in x:
            x = x[:-1]
            unit_multiplier = 1000000
        if "K" in x:
            x = x[:-1]
            unit_multiplier = 1000
        x = x.replace(",", "")
        try:
            x = float(x) * unit_multiplier
            return x
        except:
            return np.nan
    else:
        return float(x)

def get_quantile_by_variable(df, ascending_sort_var, ascending_quantile_start, ascending_quantile_end, vars_to_describe):

    df.sort_values(ascending_sort_var, ascending=True, inplace=True)
    num_rows = df.shape[0]
    start_idx = int(ascending_quantile_start * num_rows)
    end_idx = int(ascending_quantile_end * num_rows)
    quantile_df = df.iloc[start_idx:end_idx, :]
    return quantile_df[vars_to_describe]

def get_match_candidate_score(frequency_dict, org_name, candidate_match_name):
    if not isinstance(org_name, str):
        org_name = ""
    if not isinstance(candidate_match_name, str):
        candidate_match_name = ""
    
    if not org_name or not candidate_match_name:
        return 0.0
    
    org_tokens = org_name.split(' ')
    
    # tokenize the candidate match
    candidate_match_tokens = set(candidate_match_name.split(" "))

    max_dist = 1

    # Calculate the match score
    total_inverse_frequency = 0
    total_matching_inverse_frequency = 0
    tokenized_name = org_tokens
    for token in tokenized_name:
        token_frequency = frequency_dict.get(token, 999999) # if token not found, give high frequency to ignore it
        token_inverse_frequency = 1.0/token_frequency
        total_inverse_frequency += token_inverse_frequency

        best_token_similarity = 0.0
        
        for candidate_token in candidate_match_tokens:
            if not token or not candidate_token:
                continue

            dist = damerau_levenshtein_distance(token, candidate_token)

            if dist <= max_dist:
                max_len = max(len(token), len(candidate_token))
                if max_len == 0: continue
                similarity = 1.0 - (dist / max_len)

                best_token_similarity = max(best_token_similarity, similarity)
        if best_token_similarity > 0.0:
            total_matching_inverse_frequency += token_inverse_frequency * best_token_similarity
    
    try:
        match_score = total_matching_inverse_frequency / total_inverse_frequency
    except ZeroDivisionError:
        match_score = 0.0

    #Multiplicative DL Penalty
    m = len(org_name)
    n = len(candidate_match_name)
    if m == 0 or n == 0: return 0.0

    dl_distance = damerau_levenshtein_distance(org_name, candidate_match_name)
    normalized_dl = dl_distance / max(m, n)

    final_score = match_score * (1 - normalized_dl)
    return max(0.0, final_score)


REBUILD_DATSETS = True
covariate_dfs = get_covariate_dfs()
if REBUILD_DATSETS:

    ## PART 1: Match records from the gathered organization datasets (FDIC, FFEIC, Nonprofits, CIK, Compustat, etc.) to scraped comments

    # 1.1: Read and clean org names from gathered org datasets
    #sources = ["FDIC_Institutions", "FFIECInstitutions", "CreditUnions", "CIK", "compustat_resources", "nonprofits_resources", "opensecrets_resources_jwVersion", "SEC_Institutions"]
    sources = ["FDIC_Institutions", "FFIECInstitutions", "CreditUnions", "CIK", "compustat_resources", "opensecrets_resources_jwVersion", "SEC_Institutions"]
    #sources = ["FFIECInstitutions", "CIK"]

    org_name_dict = {}
    if True:
        financial_datasets = []
        unique_ids = []
        all_org_names = []
        for financial_dataset in sources:
            intermediate_data_folder = "data/"
            col_name = ""
            read_from_file = False
            if financial_dataset == "FDIC_Institutions":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "NAME"
                read_from_file = True
            elif financial_dataset == "FFIECInstitutions":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "Financial Institution Name"
                read_from_file = True
            elif financial_dataset == "CreditUnions":
                col_name = "CU_NAME"
                read_from_file = True
            elif financial_dataset == "CIK":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "company_name"#"COMPANYNAME"
                read_from_file = True
            elif financial_dataset == "compustat_resources":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "conm"
                read_from_file = True
            elif financial_dataset == "nonprofits_resources":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "name"
                read_from_file = True
            elif financial_dataset == "opensecrets_resources_jwVersion":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "orgName"
                read_from_file = True
            elif financial_dataset == "SEC_Institutions":
                intermediate_data_folder = "data/merged_resources/"
                col_name = "Name"
                read_from_file = True

            print(financial_dataset)
            if financial_dataset == "opensecrets_resources_jwVersion":
                df = pd.read_csv(BASE_DIR + intermediate_data_folder + financial_dataset + ".csv", quotechar='"')
            else:
                df = pd.read_csv(BASE_DIR + intermediate_data_folder + financial_dataset + ".csv")
            df['unique_id'] = financial_dataset + "-" + df.index.astype(str)
            df['financial_dataset'] = financial_dataset
            financial_datasets = financial_datasets + list(df['unique_id'])
            unique_ids = unique_ids + list(df['unique_id'])
            all_org_names = all_org_names + list(df[col_name])

        data = list(zip(unique_ids, all_org_names, financial_datasets))
        org_name_df = pd.DataFrame(data, columns=['unique_id', 'org_name', 'financial_dataset'])
        # org_name_df_lst = []
        # for source in sources:
        #     org_name_df_lst.append(org_name_df[org_name_df['financial_dataset']==source]['org_name'].apply(clean_fin_org_names))
        org_name_df['original_org_name'] = org_name_df['org_name']
        org_name_df['org_name'] = org_name_df['org_name'].apply(clean_fin_org_names)


    rdata_path = BASE_DIR + "data/org_counts.RData"

    print(f"Loading organization counts from: {rdata_path}")


    try:
        rdata_results = pyreadr.read_r(rdata_path)
        df = rdata_results[list(rdata_results.keys())[0]].copy()
    except Exception as e:
        print(f"Error loading RData file: {e}. Ensure pyreadr is installed and the path is correct.")
        sys.exit(1)
# --- END: Load RData Block ---

# 1. Clean and standardize the DataFrame to match the format expected by the rest of the script.

# Check the columns in the RData file and rename them to match the script's expected columns:
try:
    if 'org_name' in df.columns:
        df.rename(columns={'org_name': 'organization'}, inplace=True)
    else:
        print("Error: RData file does not contain the expected 'org_name' column.")

    df['comment_url'] = ""
except Exception as e:
    print(f"Error loading RData file: {e}. Ensure pyreadr is installed and the path is correct.")
    sys.exit(1)

df['submitter_name'] = ""
df['agency_acronym'] = "RDATA" # Placeholder
df['docket_id'] = ""
df['comment_title'] = ""

# Select only the columns the script expects later
cols = ['comment_url', 'submitter_name', 'organization', 'agency_acronym', 'docket_id', 'comment_title']
df = df[cols]

df['original_organization_name'] = df['organization']

# The following cleaning lines should be kept to ensure consistency, 
# even if the RData data is already partially clean.
# FRS, FDIC, SEC cleaning blocks are skipped for the RData sample.

# Clean the names using your function
df['submitter_name'] = df['submitter_name'].map(clean_fin_org_names)
df['organization'] = df['organization'].map(clean_fin_org_names)

# replace none
df.loc[df['submitter_name'].isna(), "submitter_name"] = ''

key_names_list = df.iloc[:, :] # This DataFrame is now your list of organizations to match

print("Finished cleaning with RData source.")

    


    # Make a (slightly) educated guess as to the submitter_name and organization for the Fed
"""
    new_key_names_list = []
    for elem in key_names_list:
        submitter_name = elem[1]
        org_name = elem[2]
        agency_acronym = elem[3]
        # TODO: we may need to consider the names
        # if agency_acronym == "FRS":
        #     comment_title = elem[5]
        #     if "(" in comment_title:
        #         comment_title = comment_title[:comment_title.index("(")].strip()
        #     clauses = comment_title.split(",")
        #     if len(clauses) == 0:
        #         clauses = comment_title.split(";")
        #     if len(clauses) == 0:
        #         pass
        #     elif len(clauses) == 1:
        #         org_name = clean_fin_org_names(clauses[0])
        #         submitter_name = org_name
        #     else:
        #         org_name = clean_fin_org_names(clauses[0])
        #         submitter_name = clean_fin_org_names(clauses[1])
        #     print(submitter_name + " | " + org_name)
        new_key_names = (elem[0], submitter_name, org_name, agency_acronym, elem[4])
        new_key_names_list.append(new_key_names)

    key_names_list = new_key_names_list
    """
print("Finished cleaning")


    # 1.3: Create 2 dicts with frequency counts of every token in the org and submitter name fields of the scraped comments db
    # submitter_frequency_dict = {}
    # org_frequency_dict = {}
    # for _, key_name in key_names_list.iterrows():
    #     org_name = key_name[2]
    #     for token in org_name.split(" "):
    #         if token in org_frequency_dict:
    #             org_frequency_dict[token] += 1
    #         else:
    #             org_frequency_dict[token] = 1



from collections import Counter
from tqdm.auto import tqdm # Keep tqdm if you want a progress bar on the split operation
tqdm.pandas()
print('Preparing candidate frequency dictionary (Vectorized).')
# 1. Split every string in the Series into a list of tokens.
token_lists = org_name_df['org_name'].progress_apply(lambda x: x.split(" "))

# 2. Flatten the list of lists into a single list of all tokens.
all_tokens = [token for sublist in token_lists for token in sublist]

# 3. Use Counter for fast, efficient frequency counting.
candidate_frequency_dict = Counter(all_tokens)
# Create linking dataset
# 1.4: Create a dict mapping from tokens in the gathered org datasets to IDs and org_names that contain that token
candidate_match_dict = {}
print('Preparing candidate match dictionary.')
for row_idx in tqdm(range(len(org_name_df))):
    row = org_name_df.iloc[row_idx]
    unique_id = row['unique_id']
    org_name = row['org_name']
    original_org_name = row['original_org_name']
    for token in org_name.split(" "):
        if token in candidate_match_dict:
            candidate_match_dict[token].append((unique_id, org_name, original_org_name))
        else:
            candidate_match_dict[token] = [(unique_id, org_name, original_org_name)]
                
#================
#1.5.1 CASCADE MATCHING
#=================

match_dict = {}
names_matched_in_1 = []
names_matched_in_2 = []
names_matched_in_3 = []
names_matched_in_4 = []

#Create the merge key in the candidate pool pool
print("Generating search keys for candidate pool...")
org_name_df['name_lower'] = org_name_df['original_org_name'].astype(str).str.lower().str.strip()

#STAGE 1: EXACT IDENTITY MATCH 
print(f"Starting Stage 1: Exact Identity Matching on {len(key_names_list)} records...")
scraped_names_df = pd.DataFrame({'original_name': key_names_list['original_organization_name'].unique()})
scraped_names_df['clean_name'] = scraped_names_df['original_name'].astype(str).str.lower().str.strip()

tier1_merge = scraped_names_df.merge(
    org_name_df[['org_name', 'name_lower', 'unique_id']], 
    left_on='clean_name', 
    right_on='name_lower', 
    how='inner'
)

tier1_by_source = {}
for _, row in tier1_merge.iterrows():
    orig_name = row['original_name']
    official_name = row['org_name']

    match_dict[orig_name] = pd.DataFrame([{
        'match_score': 1.0,
        'candidate_match_name': row['org_name'],
        'original_org_name': orig_name,
        'unique_id': row['unique_id'],
    }])
    names_matched_in_1.append(orig_name)
    
    # Organize by source
    src = str(row['unique_id']).split('-')[0]
    tier1_by_source.setdefault(src, []).append({
        'scraped_uncleaned_name': orig_name, # Added
        'matched_official_name': official_name, # Added
        'unique_id': row['unique_id'], 
        'score': 1.0
    })

for src, rows in tier1_by_source.items():
    pd.DataFrame(rows).to_csv(f"matches_stage1_{src}.csv", index=False)

# Define remaining unique names for fuzzy matching
matched_set = set(names_matched_in_1)
remaining_names = [n for n in scraped_names_df['original_name'].unique() if n not in matched_set and n != ""]



#STAGE 2: FIRST-LETTER BLOCKING
print(f"Starting Stage 2: First-Letter Blocking on {len(remaining_names)} unique names...")
tier2_by_source = {}

for org_name in tqdm(remaining_names):
    if not org_name: continue
    first_char = org_name[0].upper()
    org_tokens = org_name.split(" ")
    org_token_frequencies = sorted([(t, candidate_frequency_dict.get(t, 999999)) for t in org_tokens], key=lambda x: x[1])
    
    candidate_matches = []
    if org_token_frequencies:
        for most_unique_token, _ in org_token_frequencies[:1]: 
            if most_unique_token in candidate_match_dict:
                for row in candidate_match_dict[most_unique_token]:
                    if row[1] and row[1][0].upper() == first_char:
                        score = get_match_candidate_score(candidate_frequency_dict, org_name, row[1])
                        if score > 0.10:
                            candidate_matches.append((score, row[1], row[2], row[0]))

    if candidate_matches:
        candidate_matches.sort(key=lambda x: (-x[0], abs(len(x[1].split(" ")) - len(org_tokens))))
        best_score, best_official_name, _, best_id = candidate_matches[0]
        match_dict[org_name] = pd.DataFrame(candidate_matches, columns=['match_score','candidate_match_name', 'original_org_name', 'unique_id'])
        src = str(best_id).split('-')[0]
        tier2_by_source.setdefault(src, []).append({
        'scraped_uncleaned_name': org_name,     # The messy name
        'matched_official_name': best_official_name, # The official library name
        'unique_id': best_id, 
        'score': best_score
        })

for src, rows in tier2_by_source.items():
    pd.DataFrame(rows).to_csv(f"matches_stage2_{src}.csv", index=False)

#STAGE 3: FLEXIBLE FIRST-LETTER
matched_set.update(names_matched_in_2)
remaining_names = [n for n in remaining_names if n not in matched_set]
print(f"Starting Stage 3: Flexible Match on {len(remaining_names)} names...")
tier3_by_source = {}

for org_name in tqdm(remaining_names):
    if len(org_name) < 4: continue
    org_tokens = org_name.split(" ")
    scraped_first_letters = {t[0].upper() for t in org_tokens if len(t) > 2}
    org_token_frequencies = sorted([(t, candidate_frequency_dict.get(t, 999999)) for t in org_tokens if len(t) > 3], key=lambda x: x[1])
    
    candidate_matches = []
    if org_token_frequencies:
        unique_token = org_token_frequencies[0][0]
        if unique_token in candidate_match_dict:
            for row in candidate_match_dict[unique_token]:
                if row[1] and row[1][0].upper() in scraped_first_letters:
                    score = get_match_candidate_score(candidate_frequency_dict, org_name, row[1])
                    if score > 0.10:
                        candidate_matches.append((score, row[1], row[2], row[0]))

    if candidate_matches:
        candidate_matches.sort(key=lambda x: (-x[0], abs(len(x[1].split(" ")) - len(org_tokens))))
        best_score, best_official_name, _, best_id = candidate_matches[0]
        match_dict[org_name] = pd.DataFrame(candidate_matches, columns=['match_score', 'candidate_match_name', 'original_org_name', 'unique_id'])
        src = str(best_id).split('-')[0]
        tier3_by_source.setdefault(src, []).append({
        'scraped_uncleaned_name': org_name,     # The messy name
        'matched_official_name': best_official_name, # The official library name
        'unique_id': best_id, 
        'score': best_score
        })

for src, rows in tier3_by_source.items():
    pd.DataFrame(rows).to_csv(f"matches_stage3_{src}.csv", index=False)

#STAGE 4: RELAXED FUZZY
matched_set.update(names_matched_in_3)
remaining_names = [n for n in remaining_names if n not in matched_set]
print(f"Starting Stage 4: Relaxed Fuzzy (Threshold 0.70) on {len(remaining_names)} names...")
tier4_by_source = {}

for org_name in tqdm(remaining_names):
    if len(org_name) < 4: continue # Slightly more inclusive than < 5
    
    org_tokens = org_name.split(" ")
    # Filter out stopwords and very short tokens to find meaningful search terms
    org_token_frequencies = sorted([
        (t, candidate_frequency_dict.get(t, 999999)) 
        for t in org_tokens if t not in STOPWORDS and len(t) > 2
    ], key=lambda x: x[1])
    
    candidate_matches = []
    # Search using the top 3 unique tokens for better recall
    for unique_token, _ in org_token_frequencies[:3]: 
        if unique_token in candidate_match_dict:
            for row in candidate_match_dict[unique_token]:
                score = get_match_candidate_score(candidate_frequency_dict, org_name, row[1])
                
                # LOWERED THRESHOLD: 0.70 allows for more fuzzy variance
                if score > 0.10:
                    candidate_matches.append((score, row[1], row[2], row[0]))

    if candidate_matches:
        # Sort by score, then by how similar the word counts are
        candidate_matches.sort(key=lambda x: (-x[0], abs(len(x[1].split(" ")) - len(org_tokens))))
        best_score, best_official_name, _, best_id = candidate_matches[0]
        match_dict[org_name] = pd.DataFrame(candidate_matches, columns=['match_score', 'candidate_match_name', 'original_org_name', 'unique_id'])
        src = str(best_id).split('-')[0]
        tier4_by_source.setdefault(src, []).append({
        'scraped_uncleaned_name': org_name,     # The messy name
        'matched_official_name': best_official_name, # The official library name
        'unique_id': best_id, 
        'score': best_score
        })

# Export new Stage 4 results
for src, rows in tier4_by_source.items():
    pd.DataFrame(rows).to_csv(f"matches_stage4_relaxed_{src}.csv", index=False)

# FINAL CLEANUP: Ensure every record has a match entry
print("Finalizing match dictionary for downstream processing...")
all_unique_orgs = key_names_list['organization'].unique()

for org_name in all_unique_orgs:
    if pd.isna(org_name) or org_name == "":
        continue
    if org_name not in match_dict:
        match_dict[org_name] = pd.DataFrame(columns=[
            'match_score',
            'candidate_match_name',
            'original_org_name',
            'unique_id'
        ])
print(f"Total unique organizations in match_dict: {len(match_dict)}")

# 1.5.2: Save the candidate matches and get record counts
# with open(BASE_DIR + "data/finreg_jaccard_match_" + curr_date + ".pkl", 'wb') as pkl_out:
#     pickle.dump(match_dict, pkl_out)

print("Num scraped records: " + str(len(key_names_list)))


# 1.6: Extract the scraped records with at least one candidate match and take the top top_matches_num (or all if there are < top_matches_num) matches from the scored candidate matches
# DONE: loop until we get top match from each dataset
good_matches = {}
threshold = 0.10
counter = 0
match_counter = 0
covariate_dict = {}
frs_counter = 0

print("Mapping match results to record tuples...")
for elem_tuple in tqdm(key_names_list.itertuples(index=False, name=None), total=len(key_names_list)):
    # In your RData setup, the cleaned 'organization' name is at index 2 
    # and 'original_organization_name' is at index 6 (based on your cols list)
    cleaned_name_key = elem_tuple[2]
    
    if cleaned_name_key in match_dict and not match_dict[cleaned_name_key].empty:
        df_matches = match_dict[cleaned_name_key]
        valid_matches = df_matches[df_matches['match_score'] >= threshold]
        
        if not valid_matches.empty:
            # We provide a placeholder for the NER tag data: (is_likely_org, tags)
            # This allows the loop in 1.6 to unpack correctly: matches, tag_data = ...
            good_matches[elem_tuple] = (valid_matches, (True, ["ORG"]))

print(f"Bridge complete. {len(good_matches)} records ready for assembly.")

print("Assembling final match results...")
for elem_idx, elem_tuple in tqdm(enumerate(key_names_list.itertuples(index=False, name=None)), total=len(key_names_list)):
    
    if elem_tuple not in good_matches:
        counter += 1
        continue

    match_counter += 1
    # Unpack based on your specific tuple structure
    url, submitter, cleaned_org, agency, docket, title, raw_name = elem_tuple

    matches_df, tag_data = good_matches[elem_tuple]
    is_likely_org, org_tags = tag_data

    # Pull the pre-saved names from the first match
    top_match = matches_df.iloc[0]
    matched_official_name = top_match['candidate_match_name']
    score = top_match['match_score']

    covariate_dict[elem_tuple] = {
        'original_org_name': raw_name,       # Mapped from KeyError
        'comment_org_name': cleaned_org,     # Mapped from KeyError
        'matched_official_name': matched_official_name,
        'match_score': score,
        'comment_url': url,                  # Mapped from KeyError
        'docket_id': docket,                 # Mapped from KeyError
        'comment_agency': agency,
        'num_org_matches': len(matches_df),  # Mapped from KeyError
        'is_likely_org': is_likely_org,
        'org_tags': str(org_tags)
    }
        
print("Num records in match_dict: " + str(len(match_dict)))
print("Num records without a match: " + str(counter))
print("Share of records that weren't matchable: " + str(counter / len(match_dict)))


## PART 2: Attempt to estimate whether comment was submitted by a person or an organization
nlp = en_core_web_lg.load()

# 2.1: Among the matchable scraped comment records, use spacy's ner tagger to tag the tokens in the submitter name and org name of each record. 
good_matches_org_tagged = {}
print("Starting NLP Tagging and final match mapping...")
threshold = 0.10

for elem_tuple in tqdm(key_names_list.itertuples(index=False, name=None), total=len(key_names_list)):
    # original_organization_name is typically index 6 based on your cols selection
    name_key = elem_tuple[6] 
    
    if name_key in match_dict and not match_dict[name_key].empty:
        df_matches = match_dict[name_key]
        valid_matches = df_matches[df_matches['match_score'] >= threshold]
        
        if not valid_matches.empty:
            # We skip the heavy NER loop for now and tag as ORG if matched to library
            good_matches_org_tagged[elem_tuple] = (valid_matches, (True, ["ORG"]))

#==============================================================================
# FINAL ASSEMBLY LOOP
#==============================================================================

covariate_dict = {}
match_counter = 0

print("Assembling final match results...")
for elem_idx, elem_tuple in tqdm(enumerate(key_names_list.itertuples(index=False, name=None)), total=len(key_names_list)):
    
    if elem_tuple not in good_matches_org_tagged:
        continue

    # Unpack the tuple: [url, submitter, organization, agency, docket, title, original_name]
    _, _, _, agency, _, _, raw_name = elem_tuple

    matches_df, tag_data = good_matches_org_tagged[elem_tuple]
    is_likely_org, org_tags = tag_data

    # Extract readable names from the top match
    top_match = matches_df.iloc[0]
    matched_official_name = top_match['candidate_match_name']
    score = top_match['match_score']
    match_id = top_match['unique_id']

    # Build the record
    covariate_dict[elem_tuple] = {
        'original_org_name': raw_name,
        'matched_official_name': matched_official_name,
        'unique_id': match_id,
        'match_score': score,
        'comment_agency': agency,
        'is_likely_org': is_likely_org,
        'org_tags': str(org_tags),
        'num_org_matches': len(matches_df),  # Added this so it doesn't go missing
        'comment_org_name': _,
    }
    match_counter += 1

print(f"Assembly complete. Total matches: {match_counter}")

print("FRS counter: " + str(frs_counter))
print("Finished creating data dicts")

variables = set()
for elem_idx, elem in tqdm(enumerate(covariate_dict)):
    variables = variables.union(set(covariate_dict[elem].keys()))
variables = list(variables)
variables.sort()
print("Finished establishing variables")

data = []
for elem_idx, elem in tqdm(enumerate(covariate_dict)):
    elem_data_dict = covariate_dict[elem]
    elem_data_row = [None]*len(variables)
    for var_idx, variable in enumerate(variables):
        if variable in elem_data_dict:
            elem_data_row[var_idx] = elem_data_dict[variable]
    data.append(elem_data_row)
    # if elem_idx % 10000 == 0:
    #     print(elem_idx)
print("Finished creating items for df")

covariate_df = pd.DataFrame(list(covariate_dict.values()))
print("Columns currently in df:", covariate_df.columns.tolist())

# 3.3: Save the dataframe of scraped records with attached covariates

# filter columns
common_tails = ['best_match_name',
                'original_match_name', 
                'best_match_score', 
                'CIK', 
                'CU_NUMBER', 
                'RSSD', 
                'CERT', 
                'FED_RSSD',
                'FDIC Certificate Number',
                'IDRSSD',
                'OCC Charter Number',
                'SIC',
                'Ticker',
                'cik',
                'cusip',
                'gvkey',
                'naics',
                'sic',
                'tic',
                'ein',
                'name',
                'parentID'
                ]
desired_cols = [
    'original_org_name',
    'num_org_matches', 
    'comment_agency',
    'comment_org_name',
    'matched_official_name', # Added this - it's your most important result!
    'match_score',           # Added this - so you know how good the match is
    'docket_id',
    'comment_url',
    'unique_id',
    'is_likely_org'
]

# 2. Add columns that match our 'common_tails' (the financial data from FDIC/SEC/etc)
# We use a list comprehension that checks if the column exists in covariate_df first
final_cols = [x for x in covariate_df.columns if (x.split(':')[-1] in common_tails) or (x in desired_cols)]

# 3. SAFETY CHECK: Filter the list to only include columns that are actually in the dataframe
# This prevents the KeyError!
existing_final_cols = [c for c in final_cols if c in covariate_df.columns]

# 4. Final selection and reordering
covariate_df = covariate_df[existing_final_cols]

# Optional: Reorder so that non-technical columns come first, and data with ":" comes last
cols = covariate_df.columns
ordered_cols = [x for x in cols if not ':' in x] + [x for x in cols if ':' in x]
covariate_df = covariate_df[ordered_cols]

print(f"Final DataFrame shape: {covariate_df.shape}")


# write df
with open(BASE_DIR + "data/finreg_commenter_covariates_df_" + curr_date + ".pkl", 'wb') as pkl_out:
    pickle.dump(covariate_df, pkl_out)

covariate_df.to_csv(BASE_DIR + "data/finreg_commenter_covariates_df_" + curr_date + ".csv")

df = covariate_df
df = df[list(filter(lambda x: not "submitter" in x,df.columns))]
# df = df[df['comment_org_name']!='']
df.to_csv(BASE_DIR + "data/match_data/match_all_covariates_df_" + curr_date + ".csv")

df = pd.read_csv(BASE_DIR + "data/match_data/match_all_covariates_df_" + curr_date + ".csv")
df = df.drop("Unnamed: 0", axis=1)

#isolating low scores
score_cols = [col for col in df.columns if "best_match_score" in col]
#df['max_match_score'] = df[score_cols].fillna(-100).max(axis = 1)

#LOWER_BOUND = 0.50
#UPPER_BOUND = 0.60

# Filter for scores in the moderate range and ensures the score is positive
#filter_condition = (df['max_match_score'] > LOWER_BOUND) & \
#                    (df['max_match_score'] < UPPER_BOUND) & \
#                    (df['max_match_score'] > 0)

#df = df[filter_condition].copy()

#print(f"Filtered DataFrame down to {len(df)} records ( {LOWER_BOUND} < score < {UPPER_BOUND}).")

#Add Exact Match column
name_cols = list(filter(lambda x: "best_match_name" in x,df.columns))
exact_matches = pd.DataFrame()
for name_col in name_cols:
    exact_matches[name_col] = df[name_col]==df['comment_org_name']

new_col = (exact_matches.sum(axis=1)>0).astype(int)
df['exact_match_present'] = new_col



#final_filename = f"match_df_moderate_sample_{int(LOWER_BOUND*100)}_{int(UPPER_BOUND*100)}_" + curr_date + ".csv"
final_filename = f"match_df_full_sample_" + curr_date + ".csv"
df.to_csv(BASE_DIR + "data/match_data/" + final_filename, index=False)
print(f"Saved moderate sample to: {final_filename}")

#compare with hand matches
print("Running automated accuracy check...")
try:
    hand_match_path = BASE_DIR + "data/hand_match.csv"
    hand_df = pd.read_csv(hand_match_path) 
 
    comparison = df.merge(
        hand_df, 
        on='original_org_name', 
        how='inner', 
        suffixes=('_script', '_hand')
    )
    
    if not comparison.empty:

        hand_col = 'hand_match' 
        
        # matches FDIC?
        script_col = 'FDIC_Institutions-orgMatch:best_match_name'
        
        if script_col in comparison.columns and hand_col in comparison.columns:
            # Clean strings to avoid mismatches due to casing or whitespace
            comparison['match_correct'] = (
                comparison[script_col].astype(str).str.lower().str.strip() == 
                comparison[hand_col].astype(str).str.lower().str.strip()
            )
        
            accuracy = comparison['match_correct'].mean()
            print(f"Accuracy Check Complete!")
            print(f"Found {len(comparison)} overlapping records.")
            print(f"Script Accuracy: {accuracy:.1%}")
        
            # Save the detailed report
            report_path = BASE_DIR + "data/accuracy_comparison_report.csv"
            comparison.to_csv(report_path, index=False)
            print(f"Detailed report saved to: {report_path}")
        else:
            print(f"Missing columns for comparison. Available: {list(comparison.columns)}")
            
    else:
        print("No overlapping records found. Check if 'original_org_name' matches in both files.")

except Exception as e:
    print(f"Could not run evaluation: {e}")

def save_tier_to_csv(name_list, filename):
    results = []
    for name in name_list:
        if name in match_dict and not match_dict[name].empty:
            top_match = match_dict[name].iloc[0].copy()
            top_match['scraped_name'] = name
            results.append(top_match)
    
    if results:
        pd.DataFrame(results).to_csv(filename, index=False)
        print(f"Saved {len(results)} rows to {filename}")
    else:
        print(f"No matches found for {filename}, skipping save.")

# Execute the saves
save_tier_to_csv(names_matched_in_1, "matches_tier1_exact.csv")
save_tier_to_csv(remaining_names, "matches_fuzzy_attempts.csv")