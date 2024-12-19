import pandas as pd
import numpy as np
import json
from pprint import pprint, pformat
from copy import copy, deepcopy
import re
import random
from collections import defaultdict
import humanize
from num2words import num2words
from hashlib import md5
from encodings import utf_8
import argparse
import pydoc
SEED = 99999
random.seed(SEED)
MAX_RET = 100 # Only retrieve a maximum of this much docs
DEBUG=False

# Read the data and make dict of plausible numbers for each attribute 
df = pd.read_csv('tmdb-5000-movies.csv.gz', compression='gzip')
df['release_date'] = pd.to_datetime(df['release_date'])
ndf = deepcopy(df[['id','title','budget','revenue','release_date','popularity','runtime','production_companies','production_countries','vote_average']])
ndf['production_companies_names'] = ndf.apply(lambda row: ', '.join([comp['name'] for comp in json.loads(row.production_companies)]), axis=1)
ndf['production_countries_names'] = ndf.apply(lambda row: ', '.join([coun['name'] for coun in json.loads(row.production_countries)]), axis=1)
ndf['no_production_companies'] = ndf.apply(lambda row: len(json.loads(row['production_companies'])), axis=1)
ndf['no_production_countries'] = ndf.apply(lambda row: len(json.loads(row['production_countries'])), axis=1)
ndf = ndf.drop(['production_companies','production_countries'], axis=1)
ndf = ndf.dropna()
ndf = ndf.drop(index=ndf[(ndf['budget']==0) | (ndf['revenue']==0)].index)

Deltas_def = {
    'revenue': int(10e3),
    'budget': int(10e3),
    'runtime': 1,
    'release_date': pd.Timedelta(1,'D'),
    'popularity': 0.1,
    'vote_average': 0.5,
    'no_production_companies': 1,
    'no_production_countries': 1,
}
ssize_def = 100
def collect_numbers(Deltas=Deltas_def, uniform=False, sample_size=ssize_def):
    numbers = dict()
    if uniform:
        numbers['revenue'] = lambda : random.randrange(ndf.loc[ndf['revenue']>0, 'revenue'].min(),ndf.loc[ndf['revenue']>0, 'revenue'].max(),Deltas['revenue'])
        numbers['budget'] = lambda : random.randrange(ndf.loc[ndf['budget']>0, 'budget'].min(),ndf.loc[ndf['budget']>0, 'budget'].max(),Deltas['budget'])
        numbers['runtime'] = lambda : random.randrange(ndf.loc[ndf['runtime']>0, 'runtime'].astype(int).min(),ndf.loc[ndf['runtime']>0, 'runtime'].astype(int).max(),Deltas['runtime'])
        numbers['release_date'] = lambda : pd.to_datetime(random.randrange(ndf['release_date'].min().value,ndf['release_date'].max().value,Deltas['release_date'].value))
        numbers['popularity'] = lambda : random.randrange(int(ndf.loc[ndf['popularity']>0, 'popularity'].min()*100),int(ndf.loc[ndf['popularity']>0, 'popularity'].max()*100),int(Deltas['popularity']*100))/100
        numbers['vote_average'] = lambda : random.randrange(int(ndf.loc[ndf['vote_average']>0, 'vote_average'].min()*100),int(ndf.loc[ndf['vote_average']>0, 'vote_average'].max()*100),int(Deltas['vote_average']*100))/100
        numbers['no_production_companies'] = lambda : random.randrange(ndf.loc[ndf['no_production_companies']>0, 'no_production_companies'].min(),ndf.loc[ndf['no_production_companies']>0, 'no_production_companies'].max(),Deltas['no_production_companies'])
        numbers['no_production_countries'] = lambda : random.randrange(ndf.loc[ndf['no_production_countries']>0, 'no_production_countries'].min(),ndf.loc[ndf['no_production_countries']>0, 'no_production_countries'].max(),Deltas['no_production_countries'])
        if DEBUG: 
            print('calling fn in numbers ...')
            for ccat in numbers.keys(): print(ccat, ' : \t', numbers[ccat](), numbers[ccat](), numbers[ccat](), numbers[ccat](), numbers[ccat]())
        return numbers
    for cat in ndf.columns[2:]:
        if cat in ['production_companies_names', 'production_countries_names']: continue
        nonzro = ((ndf[cat]>0) & (pd.notna(ndf[cat]))) if cat != 'release_date' else ndf.index
        numbers[cat] = random.choices(list(ndf.loc[nonzro, cat]), k=sample_size)
        # numbers[cat] = list(pd.Series(numbers[cat])[pd.notna(numbers[cat])]) if cat != 'release_date' else numbers[cat]
    if DEBUG: print('numbers is ----\n', numbers)
    return numbers

# <<>> position numbering of numerical entities
def numb_positions(q_list):
    for query in q_list:
        line = query['q#']
        patt = re.compile("<<[\w\s\W]*?>>")
        fdfd = patt.finditer(line)
        markers = []
        for i, mm in enumerate(fdfd):
            markers.append((mm.span()[0]-i*4 , mm.span()[1]-(i+1)*4))
            query['nums'][i]['startposition'] , query['nums'][i]['endposition'] = markers[i]
def display_q(query):
    highlight = [' ']*len(query['q#'])
    for n in query['nums']:
        a, b = n['startposition'], n['endposition']
        highlight[a:b] = ['^']*(b-a)
    disp_text = query['q']+'\n'
    disp_text += ''.join(highlight)
    return disp_text


# Helper function to generate human-readable values
def generate_human_readable(value, attribute, oprn):
    try: 
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            options = [
                ("year %Y", 'unity', 'year', str(value.year)), ("%Y", 'unity', 'year', str(value.year)), ("%B %Y", 'unity', 'month', value.strftime('%Y-%m')),
                ("early %Y", 'early', 'year', str(value.year)), ("late %Y", 'late', 'year', str(value.year)), ("mid %Y", 'mid', 'year', str(value.year)), (f"{value.year // 10 * 10}s", 'unity', 'decade', str(value.year//10*10)),
                ("early '%y", 'early', 'year', str(value.year)), ("late '%y", 'late', 'year', str(value.year)), ("mid '%y", 'mid', 'year', str(value.year)),
                ("autumn %Y", 'late', 'year', str(value.year)), ("spring %Y", 'early', 'year', str(value.year)), ("%D", 'unity', 'day', value.strftime('%Y-%m-%d')),
                ('%b %Y', 'unity', 'month', value.strftime('%Y-%m')), ('%d %B %Y', 'unity', 'day', value.strftime('%Y-%m-%d')), ('%d %b %Y', 'unity', 'day', value.strftime('%Y-%m-%d')), ('\'%y', 'unity', 'year', str(value.year)), ("'"+f"{value.year // 10 * 10}s"[-3:], 'unity', 'decade', str(value.year//10*10)),
                (f"early {value.year // 10 * 10}s", 'early', 'decade', str(value.year//10*10)), (f"late {value.year // 10 * 10}s", 'late', 'decade', str(value.year//10*10)), (f"mid {value.year // 10 * 10}s", 'mid', 'decade', str(value.year//10*10))
            ]
            format, scalefactor, unit, quantity = random.choice(options)
            humn_text = value.strftime(format=format)
            return humn_text, scalefactor, unit, quantity
        elif isinstance(value, (np.int_, np.float_, int, float)):
            if attribute in ["budget", "revenue"]:
                val_ord = np.floor(np.log10(value))
                options = []
                if val_ord < 6 :
                    prec=1
                    num0 = round(value/1000, prec)*1000
                    while num0 == 0: 
                        num0 = round(value/1000, prec)*1000
                        prec += 1
                    options.extend([
                        (humanize.intword(value, f'%.{prec}f'), 'thousand', 'dollar', float(num0)), (f'${num0/1000:.{prec}f} thousand', 'thousand', 'dollar', float(num0))
                    ])
                    return random.choice(options)
                if val_ord >= 8 :
                    num2 = value // int(10e6) * int(10e6)
                    options.extend([
                        (humanize.intword(num2, '%.2f'), 'billion' if val_ord>8 else 'million', 'dollar', num2),
                        (humanize.intword(num2, '%.2f').replace('billion','B').replace('million','M'), 'billion' if val_ord>8 else 'million', 'dollar', num2),
                        (f'${value/int(1e9):.2f} B', 'billion', 'dollar', round(value/int(1e9),2)*int(1e9)), (f'{value/int(1e9):.2f}B dollar', 'billion', 'dollar', round(value/int(1e9),2)*int(1e9)), 
                    ])
                    return random.choice(options)
                num1 = value // int(1e6) * int(1e6)
                num3 = round(value/1_000_000)*1_000_000
                options.extend([
                    (humanize.intword(num1), 'million', 'dollar', num1), (humanize.intword(num1).replace("million", "M"), 'million', 'dollar', num1),
                    (f"${num3:,}", 'unity', 'dollar', num3), (f"${num3:,}".replace(",", " "), 'unity', 'dollar', num3),
                    (num2words(num3*100, to="currency", currency="USD").removesuffix(', zero cents'), 'million', 'dollar', num3),
                    (num2words(num3*100, to="currency", currency="USD").removesuffix(', zero cents').replace('million', 'M'), 'million', 'dollar', num3),
                    (f'{value/int(1e6):.2f} M', 'million', 'dollar', round(value/int(1e6),2)*int(1e6)), (f'${value/int(1e6):.2f} M', 'million', 'dollar', round(value/int(1e6),2)*int(1e6)),
                ])
                return random.choice(options)
            elif attribute == "runtime":
                options = [(f"{value} minutes", 'unity', 'minute', value), (f"{value} min", 'unity', 'minute', value), (f"{value} mins", 'unity', 'minute', value),
                        (f"{value/60:.1f} hours", 'unity', 'hour', round(value/60,1)), (f"{value/60:.1f} hr", 'unity', 'hour', round(value/60,1)), (f"{value/60:.1f} hrs", 'unity', 'hour', round(value/60,1)),
                        (num2words(round(value/60))+' hrs', 'unity', 'hour', round(value/60,1))
                ]
                return random.choice(options)
            elif attribute.endswith("companies"):
                options = [str(value)]#+(' company' if value==1 else ' companies'), num2words(value)+(' company' if value==1 else ' companies')]
                return random.choice(options), 'unity', 'company', value
            elif attribute.endswith("countries"):
                options = [str(value)]#+(' country' if value==1 else ' countries'), num2words(value)+(' country' if value==1 else ' countries')]
                return random.choice(options), 'unity', 'country', value
            elif attribute == "popularity":
                options = [f'{value:.2f}', f"{value:.2f} points"]
                return random.choice(options), "unity", "point", round(value,2)
            elif attribute == "vote_average":
                options = [f'{value:.2f}', f"{value:.2f}/10", f"{value:.2f} stars", f"{value:.2f}/10 stars"]
                return random.choice(options), "unity", "star", round(value,2)
        else:
            raise KeyError(f'{value} value unknown . attribute {attribute} , {type(value)}\n'
                           f'')
    except Exception as e:
        print('error in assigning!')
        print(f'{value} value unknown . attribute {attribute} , {type(value)}')
        raise e
    raise KeyError(f'{value} value unknown . attribute {attribute} , {type(value)}')
    return str(value), "unity", "unknown", 

# Function to randomly fill in numbers and annotate full templates
def insert_numbers(templates, numbers, random_seed=None):
    if random_seed: random.seed(random_seed)
    numified_queries = []
    
    for template in templates:
        q_template = template["q#"]
        nums_annotations = []
        
        for num in template["nums"]:
            index = num["index"]
            attr_type = num["type"]
            operation = num["operation"]
            
            # Select a random number or call a random sampler from the numbers dict for this attribute
            plausible_values = numbers[attr_type]
            if callable(plausible_values):
                selected_value = plausible_values()
            else: selected_value = random.choice(plausible_values)
            
            # Generate human-readable value
            human_readable, scalefactor, unit, quantity = generate_human_readable(selected_value, attr_type, operation)
            
            # Replace the placeholder in the template
            placeholder = f"<<>>"
            q_template = q_template.replace(placeholder, f"<<{human_readable}>>", 1)

            # Add to nums annotations
            nums_annotations.append({
                "index": index,
                "quantity": quantity,
                "scalefactor": scalefactor,
                "unit": unit,
                "type": attr_type,
                "operation": operation
            })
        
        # Append the annotated query to the final list
        numified_queries.append({
            "q": q_template.replace('<<','').replace('>>',''),
            "q#": q_template,
            "nums": nums_annotations
        })
    
    return numified_queries

# Make unique query ids
def make_digest_qids(q_list):
    for qu in q_list:
        zxz = utf_8.encode(qu['q'])
        qu['qid'] = md5(zxz[0],usedforsecurity=False).hexdigest()

def fetch_by_qid(q_list, qid):
    for q in q_list:
        if q['qid'] == qid: return q
    raise IndexError(f"No query with given id {qid} found.")

# Make the operands dict for each query
def fix_operands(q_list, alloperations):
    for query in q_list:
        query['operands'] = {}
        for op in alloperations : query['operands'][op]=[]
        temp=[]
        for n in query['nums']:
            if n['operation'] == 'between':
                if len(temp)<2: temp.append(n['index'])
                if len(temp) == 2:
                    query['operands']['between'].append(temp)
                    temp = []
            else: query['operands'][n['operation']].append(n['index'])

thresholds_def = {          # Default threshold values, used for scaling rank scores
        'revenue': int(100e6),
        'budget': int(20e6),
        'runtime': 10,
        'release_date': pd.Timedelta(50,'W'),
        'popularity': 10,
        'vote_average': 3,
        'no_production_companies': 2,
        'no_production_countries': 2,
    }
cutoffs_def = {
        'revenue': int(700e6),                  # not more than 700M away
        'budget': int(70e6),                    # not more than 70M away
        'runtime': 30,                          # not more than 30 minutes away
        'release_date': pd.Timedelta(1040,'W'), # not more than 20 years ago/later
        'popularity': 50,                       # not more than 50 points away
        'vote_average': 8,                      # not more than 8 stars away
        'no_production_companies': 9,           # not more than 9 numbers away
        'no_production_countries': 7,           # not more than 7 numbers away
    }
# Fetch and Rank the correct documents from the annotations
def r_and_r_docs(query_list, numbers_df, thresholds=thresholds_def, cutoffs=cutoffs_def, debug=DEBUG):
    def around_(x):
        return abs(x-targetval) <= thresholds[type_]
    def around_rk(x):
        return abs(x-targetval) / (thresholds[type_] or thresholds_def[type_])
    def lessthan_(x):
        return -cutoffs[type_]*grabf<=x-targetval<=thresholds[type_]
    def lessthan_rk(x):
        return (np.vectorize(lambda x: -(x-targetval)/(cutoffs[type_]*grabf) if x<=targetval else 1+(x-targetval)/(cutoffs[type_]*grabf)))(x)
    def greaterthan_(x):
        return  cutoffs[type_]*grabf>=x-targetval>=-thresholds[type_]
    def greaterthan_rk(x):
        return (np.vectorize(lambda x:  (x-targetval)/(cutoffs[type_]*grabf) if x>=targetval else 1-(x-targetval)/(cutoffs[type_]*grabf)))(x)
    def atmost_(x):
        return -cutoffs[type_]*grabf+targetval<=x<=targetval
    def atmost_rk(x):
        return (np.vectorize(lambda x:  (x-targetval)/(cutoffs[type_]*grabf)))(x)
    def atleast_(x):
        return  cutoffs[type_]*grabf+targetval>=x>=targetval
    def atleast_rk(x):
        return (np.vectorize(lambda x: -(x-targetval)/(cutoffs[type_]*grabf)))(x)
    def exact_(x):
        return targetval[0]<=x<=targetval[1] if isinstance(targetval,tuple) else x==targetval
    def exact_rk(x):
        return np.vectorize(lambda x: 1)(x)
    def between_(x):
        return targetval[0]<=x<=targetval[1]
    def between_rk(x):
        return np.vectorize(lambda x: 1)(x)
    def in_(x):
        return targetval[0]<=x<=targetval[1]
    def in_rk(x):
        return np.vectorize(lambda x: 1)(x)
    operations_filter = {
        'around': around_ ,
        'lessthan': lessthan_ ,
        'greaterthan': greaterthan_ ,
        'exact': exact_ ,
        'between': between_,
        'in': in_ ,
        'atleast':atleast_,
        'atmost':atmost_,
    }

    operations_ranking = {
        'around': around_rk,
        'lessthan': lessthan_rk,
        'greaterthan': greaterthan_rk,
        'exact': exact_rk,
        'between': between_rk,
        'in': in_rk,
        'atleast': atleast_rk,
        'atmost': atmost_rk,
    }

    def extract_ps(numms):
        targetval = numms['quantity']
        scalefactor = numms['scalefactor']
        unit = numms['unit']
        type_ = numms['type']
        operation = numms['operation']
        if type_ == 'release_date':
            assert isinstance(targetval, str), f'dates not in string form | qid={query["qid"]} , query=\'{query["q"]}\''
            tspan = {'day':(1,'D'),'half_day':(0.5,'D'),'month':(30,'D'),'half_month':(15,'D'),'year':(365,'D'),'half_year':(182.5,'D'),'decade':(521,'W'),'half_decade':(260,'W')}
            targetval_l, targetval_r = pd.to_datetime(targetval), pd.to_datetime(targetval)+pd.Timedelta(*tspan[unit])
            if scalefactor == 'late':
                targetval_l, targetval_r = pd.to_datetime(targetval)+pd.Timedelta(*tspan['half_'+unit]), pd.to_datetime(targetval)+pd.Timedelta(*tspan[unit])
            elif scalefactor == 'early':
                targetval_l, targetval_r = pd.to_datetime(targetval), pd.to_datetime(targetval)+pd.Timedelta(*tspan['half_'+unit])
            elif scalefactor == 'mid':
                targetval_l, targetval_r = pd.to_datetime(targetval)+pd.Timedelta(*tspan[unit])*0.333, pd.to_datetime(targetval)+pd.Timedelta(*tspan[unit])*0.666
            if operation in ['lessthan','atleast']:
                targetval = targetval_l
            elif operation in ['greaterthan','atmost']:
                targetval = targetval_r
            elif operation == 'around':
                targetval = targetval_l+(targetval_r-targetval_l)/2
            elif operation in ['in', 'exact', 'between']:
                targetval = (targetval_l,targetval_r)
        if type_ == 'runtime':
            if unit in ['hours','hrs','hour','hr'] : targetval = targetval*60
        return targetval, scalefactor, unit, type_, operation
    def setparameters(numms):
        if isinstance(numms, (list, np.ndarray)) and len(numms)==2:
            targetval_1, *tv1 = extract_ps(numms[0])
            targetval_2, *tv2 = extract_ps(numms[1])
            if isinstance(targetval_1, tuple) and isinstance(targetval_2, tuple) :
                tar_low, tar_high = min(*targetval_1, *targetval_2), max(*targetval_1, *targetval_2)
            else : tar_low, tar_high = min(targetval_1, targetval_2), max(targetval_1, targetval_2)
            return (tar_low,tar_high), *tv1
        elif isinstance(numms, dict):
            return extract_ps(numms)

    q_ret_pairs = []
    for query in query_list:
        if debug: print('Query=',query['q'], '\nqid=',query['qid'])
        indiretr = []
        combret = pd.DataFrame()
        grabf = 1.0
        while len(combret)==0 and grabf<=2: 
            for opr in query['operands']:
                for nindx in query['operands'][opr]:
                    if nindx == []: continue
                    numb = np.array(query['nums'])[np.array(nindx)-1]
                    targetval, scalefactor, unit, type_, operation = setparameters(numb)
                    if debug: print(targetval, scalefactor, unit, type_, operation)
                    assert operation == opr , f"Mismatch between operand list and operation for number id {nindx} in question id {query['qid']}"
                    try:
                        rter = numbers_df.loc[numbers_df[type_].apply(operations_filter[operation]),['id','title',type_]]
                        if len(rter) > 0:
                            rter = rter.sort_values(type_, ascending=True, key=operations_ranking[operation])
                    except: 
                        raise Exception(f"{targetval, scalefactor, unit, type_, operation}")
                    rter['score'] = rter[type_].apply(operations_ranking[operation])
                    indiretr.append(rter)
                    if debug: print(rter)

            assert len(indiretr)>0 , "Empty retrieve docs. " + f"q={query['q']}\nnums={query['nums']} \noperands={query['operands']}"
            if debug: print(f"No. of independent table rows fetched: {len(indiretr)}")
            combret = indiretr.pop()
            if len(combret)==0: 
                grabf += 0.25
                if debug: print('!'*30+f'\nRedoing with grabf={grabf} because empty set')
            while len(indiretr) > 0:
                nextret = indiretr.pop()
                combret = pd.merge(combret, nextret, how='inner', on=['id','title'], suffixes=('_L','_R'))
                if debug: print(f"{targetval, scalefactor, unit, type_, operation} ")
                if debug: print(combret)
                if len(combret)==0: 
                    grabf += 0.25
                    if debug: print('!'*30+f'\nRedoing with grabf={grabf} because empty set')
                    break
                combret['score'] = combret['score_L']+combret['score_R']
                combret = combret.drop(columns=['score_L','score_R'])
        if len(combret)==0: 
            q_ret_pairs.append({'q':query['q'], 'qid':query['qid'], 'rrdf':combret})
            continue
        combret = combret.sort_values(by='score').drop(columns='score')
        if len(combret)>MAX_RET: combret = combret.iloc[:MAX_RET,:]
        q_ret_pairs.append({'q':query['q'], 'qid':query['qid'], 'rrdf':combret})
        # if debug: break
    return q_ret_pairs

def inspect_elements(q_list):
    print("Sample:")
    random_i = random.randint(0,len(q_list))
    print(json.dumps(q_list[random_i], indent=4))
    random_h_i = input("Enter random index to inspect or simply press <enter>. Press 'q' to continue : ")
    while random_h_i != 'q':
        if not random_h_i.isnumeric():
            random_h_i = random.randint(0,len(q_list))
        print(json.dumps(q_list[random_h_i], indent=4))
        random_h_i = input("Enter random index to inspect or simply press <enter>. Press 'q' to continue : ")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, required=True, help="the file containing empty queries")
    parser.add_argument("-o", "--output", type=str, required=True, help="path/name to output the constructed queries")
    parser.add_argument("-v", "--debug", action='store_true')
    parser.add_argument("-r", "--repeat", type=int, default=0, help="number of time to iterate over same templates file")
    parser.add_argument("-c", "--config", type=str, help="file containing custom parameters")
    parser.add_argument("--randomseed", type=int, help="seed value for the whole process")

    args = parser.parse_args()
    DEBUG = args.debug or DEBUG
    SEED = args.randomseed or SEED
    random.seed(SEED)

    # Set parameters
    Deltas = Deltas_def
    thresholds = thresholds_def
    cutoffs = cutoffs_def
    uniform = False
    ssize = ssize_def
    # Set custom config parameters if given
    if args.config != None :
        with open(args.config,'r') as c:
            config = json.load(c)
        Deltas = config['Deltas'] or Deltas
        thresholds = config['thresholds'] or thresholds
        cutoffs = config['cutoffs'] or cutoffs
        uniform = config['uniform'] or uniform
        ssize = config['ssize'] or ssize
            
    # Construct numbes dict accordingly
    numbers = collect_numbers(Deltas, uniform=uniform, sample_size=ssize)

    # Open the file that has queries templates with basic annotation
    print(f"Opening file {args.filename} with basic annotation template...")
    with open(args.filename, 'r') as f:
        empty_templates = json.load(f)
    # and fill in with plausible numbers randomly and annotate them fully 
    # repeat given number of times for varieties

    for iterationno in range(args.repeat+1):
        numified_queries = insert_numbers(empty_templates, numbers, random_seed=SEED)
        print(f'No. of queries constructed: {len(numified_queries)}\n'+'-'*70)

        numb_positions(numified_queries)
        print("Position marked.\n"+'-'*70)

        make_digest_qids(numified_queries)
        print("Query ids constructed.\n"+'-'*70)

        all_operations = ['exact','around','between','in','greaterthan','lessthan','atleast','atmost']
        fix_operands(numified_queries, all_operations)
        print(f"Operands defined for operations: {all_operations} \n"+'-'*70)

        # Review query sample
        wait = input('see queries...')
        for q in numified_queries:
            for n in q['nums']:
                if isinstance(n['quantity'], np.int_): n['quantity'] = int(n['quantity'])
        inspecttext = "Inspect Queries with numbers replaced : \n"
        inspecttext += f"Total no. of queries : {len(numified_queries)}\n\n"
        inspecttext += json.dumps(numified_queries, indent=4) + '\n\n'
        inspecttext += 'Queries with position markers:\n'
        for q in numified_queries: inspecttext += display_q(q) + '\n'
        pydoc.pager(inspecttext)
        # inspect_elements(numified_queries)

        print('-'*70+"\nGoing to retrieve and rank docs for each query...")
        q_a_docs = r_and_r_docs(numified_queries, ndf, thresholds, cutoffs, debug=DEBUG)
        print("Done !")
        countempty = 0
        empty_rrdfs = []
        for i, QR in enumerate(q_a_docs):
            if len(QR['rrdf'])==0:
                countempty += 1
                empty_rrdfs.append((i,QR['qid']))
        print(f'No. of empty docs: {countempty}')
        print(f'ids of queries with no retrieved docs: ')
        pprint(empty_rrdfs)
        # Review query-doc pairs
        text = '#'*130+'\n          R&R docs:\n'+'#'*130+'\n'+\
                    pformat(q_a_docs, indent=4, width=150, sort_dicts=False)
        pydoc.pager(text)

        file_to_write_final = 'run'+str(iterationno)+'_'+args.output
        confmtn = input(f"Confirm writing to file '{file_to_write_final}' or rename?  [y|N] : ")
        if confmtn == 'N':
            file_to_write_final = input("Enter filename to save final queries-docs pairs: ") or file_to_write_final
            confmtn = input(f"Confirm writing to file '{file_to_write_final}'. [y|N] : ")
        if confmtn == 'y':
            for qrd, qd in zip(numified_queries,q_a_docs): qrd['retindxs'] = list(qd['rrdf']['id'])
            for q in numified_queries:
                for n in q['nums']:
                    if isinstance(n['quantity'], np.int64): n['quantity'] = int(n['quantity'])
            with open(file_to_write_final, '+w') as f:
                json.dump(numified_queries, f, indent=4,)
            print(f"Written to file {file_to_write_final}")
