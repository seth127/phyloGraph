import requests
import collections
import json
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine, pdist, squareform
from sklearn.manifold import MDS


import webbrowser
import wikipedia
from bs4 import BeautifulSoup

import plotly
import plotly.plotly as py
from plotly.offline import iplot
import plotly.graph_objs as go

# AAA
from time import time

# constants
TREE_OF_LIFE = "http://tolweb.org/onlinecontributors/app?service=external&page=xml/"
TOL_SEARCH = "GroupSearchService&group={}"
TOL_FETCH = "TreeStructureService&node_id={}"
LIMIT = 100000 # limit on how many lines to fetch from Tree of Life
GEO_TIME = 'geological_time.csv'

def split_stat(n, start, end):
    try:
        this_stat = n.split(start)[1].split(end)[0]
    except:
        this_stat = "no {}".format(start)
    return this_stat

def rn():
    return np.random.uniform(-1,1,1)[0]

def trynum(n):
    try:
        return float(n)
    except:
        return None

def get_dates_and_cf(this_query, gt):
    """"""
    try:
        # get wikipedia page
        this_page_id = wikipedia.search(this_query, results=1)[0]
        wiki_url = "https://en.wikipedia.org/wiki/{}".format(this_page_id)
        res = requests.get(wiki_url).content.decode('utf-8')
        soup = BeautifulSoup(res, "lxml")
        
        # get infobox
        table = soup.find("table", {"class":"infobox biota"})

        # get classification text
        rows = table.findAll(lambda tag: tag.name=='td')
        cf_text = [re.sub("[^A-Za-z]+", " ", td.text).lower() for td in rows]
        cf_text = ' '.join(cf_text + [this_query.lower()])

        # get dates section
        divs = table.findAll(lambda tag: tag.name=='div')
        date_text = ' '.join([div.text.replace('\n', ' ') for div in divs])
        date_text = re.sub("\[[0-9]\]", "", date_text) # remove links
        useful = re.sub("[^A-Za-z0-9]", " ", date_text) # remove other stuff
        useful = useful.replace("Late ","Late_").replace("Middle ","Middle_").replace("Early ", "Early_").split(" ")

        # parse dates
        times = [trynum(n) for n in useful]
        times = [n for n in times if n is not None]
        if len(times) >= 2:
            d = (np.round(max(times),3), np.round(min(times),3))
        else:
            # parse from periods
            dates = []
            for u in useful:
                ru = u.replace("Late_","").replace("Middle_","").replace("Early_", "")
                try:
                    this_row = list(gt[gt['Period']==ru].iloc[0])
                    if "Late_" in u:
                        this_dates = [this_row[2]]
                    elif "Middle_" in u:
                        this_dates = [np.mean(this_row[1:3])]
                    elif "Early_" in u:
                        this_dates = [this_row[1]]
                    else:
                        this_dates = this_row[1:3]
                    dates += this_dates
                except:
                    pass
            d = (np.round(max(dates),3), np.round(min(dates),3))

    except:
        d = (None, None)
        cf_text = ''
    # return output
    return d, cf_text

def get_dates(this_query, gt):
    """
    DEPRECATED!!! this was my first try
    ----
    gets the earliest and latest date for an organism
    this_query - a text sting organism name
    gt - the geological time dataframe
    """
    try:
        # get wikipedia page
        this_page_id = wikipedia.search(this_query, results=1)[0]
        
        # fetch infobox
        wiki_url = 'http://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=json&titles={}&rvsection=0'.format(this_page_id)
        box_raw = json.loads(requests.get(wiki_url).content.decode('utf-8'))
        box = next(iter(box_raw['query']['pages'].values()))['revisions'][0]['*']

        # parse out fossil range
        fr = box.split("fossil_range")[-1].split('\n')[0].split('<ref')[0]
        useful = re.sub("[^A-Za-z0-9\.]+", " ", fr)
        useful = useful.replace("Late ","Late_").replace("Middle ","Middle_").replace("Early ", "Early_").split(" ")
        
        # try to get actual MYA times
        times = [trynum(n) for n in useful]
        times = [n for n in times if n is not None]
        if len(times) >= 2:
            d = (np.round(max(times),3), np.round(min(times),3))
        else:
            # parse from periods
            dates = []
            for u in useful:
                ru = u.replace("Late_","").replace("Middle_","").replace("Early_", "")
                try:
                    this_row = list(gt[gt['Period']==ru].iloc[0])
                    if "Late_" in u:
                        this_dates = [this_row[2]]
                    elif "Middle_" in u:
                        this_dates = [np.mean(this_row[1:3])]
                    elif "Early_" in u:
                        this_dates = [this_row[1]]
                    else:
                        this_dates = this_row[1:3]
                    dates += this_dates
                except:
                    pass
            d = (np.round(max(dates),3), np.round(min(dates),3))
    except:
        d = (None, None)
        box = ''
    # return output
    box_text = re.sub("[^A-Za-z]+", " ", box).lower()
    return d, box_text



class phyloData:
    '''
    either fetches data from the Tree of Life website,
    loads raw data from file,
    or loads prepared data from a files.
    '''
    def __init__(self):
        print("the data of phylo")
        self.df = None
        self.raw = None

    def search_name(self, query):
        '''
        Searches Tree of Life for a free text name match
        '''
        query_url = TREE_OF_LIFE+TOL_SEARCH.format(query)
        #print("fetching {} ...".format(query_url))
        raw_search = []
        with requests.get(query_url, stream=True) as r:
            i = 0
            for c in r.iter_lines():
                c = c.decode('utf-8')#.strip()
                i +=1
                raw_search.append(c)
                if i % 5000 == 0:
                    print(i, end=' ')
                if i > LIMIT:
                    break
        # parse and return
        df = self.parse_raw(raw_search, named=False, assign=False)
        return df[['name', 'id']]

    def fetch_tol_data(self, query, named=True, limit=LIMIT):
        '''
        Fetches tree starting at specified id from Tree of Life
        '''
        if not isinstance(query, int):
            return "Must pass integer 'id' to fetch. Use search_name() for free text search and 'id' look up."

        query_url = TREE_OF_LIFE+TOL_FETCH.format(query)
        print("fetching {} ".format(query_url), end='', flush=True)
        raw_resp = []
        with requests.get(query_url, stream=True) as r:
            i = 0
            for c in r.iter_lines():
                c = c.decode('utf-8')#.strip()
                i +=1
                raw_resp.append(c)
                if i % 10000 == 0:
                    print(i, end='', flush=True)
                elif i % 1000 == 0:
                    print('.', end='', flush=True)
                # break after hard limit
                if i > limit:
                    break

        print("\nFetched {} lines. Parsing and assigning...".format(i))
        #print("len(raw_resp) {}".format(len(raw_resp)))
        #print("raw_resp[:3] {}".format(raw_resp[:3]))
        # parse and return
        self.raw = raw_resp
        self.parse_raw(raw_resp, named=named)
        print("All done. {} organisms fetched.".format(self.df.shape[0]))

    def load_raw_file(self, raw_file, named=True):
        # load file
        with open(raw_file, 'r') as f:
            raw = f.read().split('\n')
        #print("{} lines in raw file".format(len(raw)))
        self.raw = raw
        # parse to df
        parse_raw(self.raw, named=named)
        print("All done. {} organisms fetched.".format(self.df.shape[0]))

    def write_raw_data(self, raw_file):
        if self.raw is not None:
            with open(raw_file, 'w') as f:
                f.write('\n'.join(self.raw))
            print("raw data written to {}".format(raw_file))
        else:
            print("Must load data first. Try load_raw_file() or fetch_tol_data() and then parse_raw()")
            
    def parse_raw(self, raw, named=True, assign=True):
        '''parses the raw output from Tree of Life
        or a file with that raw output in it

        if assign=False this function returns the 
        df instead of assigning them to self'''
        # get nodes
        nodes_raw = []
        for i in range(len(raw)):
            n = raw[i]
            try:
                nplus = raw[i+1]
            except:
                nplus = "<no next>"
            if "ID=" in n:
                nodes_raw.append(n.strip() + nplus.strip())
        #print("Found {} nodes.".format(len(nodes_raw)))

        # parse node stats
        nodes_list = []
        for n in nodes_raw:
            # get name
            try:
                this_name = split_stat(n, 'CDATA[', ']')
            except:
                this_name = None
            # get extinction status
            try:
                this_extinct = int(split_stat(n, 'NODE EXTINCT="', '"'))
            except:
                this_extinct = None
            # get phylesis
            try:
                this_phylesis = int(split_stat(n, 'PHYLESIS="', '"'))
            except:
                this_phylesis = None
            # get id
            try:
                this_id = int(split_stat(n, 'ID="', '"'))
            except:
                this_id = None
            # get num ancestors
            try:
                this_ancestor = int(split_stat(n, 'ANCESTORWITHPAGE="', '"'))
            except:
                this_ancestor = None
            # get num kids
            try:
                this_num_kids = int(split_stat(n, 'CHILDCOUNT="', '"'))
            except:
                this_num_kids = None
            # assign
            nodes_list.append({
                'extinct' : this_extinct,
                'id' : this_id,
                'phylesis' : this_phylesis,
                'name' : this_name,
                'ancestor' : this_ancestor,
                'num_kids' : this_num_kids
            })
        #print("{} nodes parsed to nodes_list".format(len(nodes_list)))
        #print("first node: {}".format(nodes_list[0]))

        # filter out non-named nodes
        if named:
            nodes_list = [n for n in nodes_list if 'CDATA' not in n['name']]
            #print("filtered non-named nodes. {} nodes left.".format(len(nodes_list)))

        # get node coords
        for node in nodes_list:
            node['depth'] = None
            node['x'] = None
            node['y'] = None
            
        # assign first one to depth 0
        nodes_list[0]['depth'] = 1
        nodes_list[0]['x'] = rn()
        nodes_list[0]['y'] = rn()

        # add Z coords and jitter x and y
        for i in range(len(nodes_list)):
            node = nodes_list[i]
            parent = [n for n in nodes_list if n['id'] == node['ancestor']]
            if len(parent) > 0:
                nodes_list[i]['depth'] = parent[0]['depth'] + 1
                # depth multiplier for narrowing
                this_depth = nodes_list[i]['depth']
                #if this_depth == 1:
                #    depth_mult = 0
                #else: 
                #    depth_mult = (3/this_depth)
                #depth_mult = (3/this_depth)
                depth_mult = 1
                    
                nodes_list[i]['x'] = parent[0]['x'] + (rn() * depth_mult)
                nodes_list[i]['y'] = parent[0]['y'] + (rn() * depth_mult)

        # pred df
        df = pd.DataFrame(nodes_list)
        #print("df.head():\n{}".format(df.head()))
        #print("phylesis: {}".format(collections.Counter(df['phylesis'])))
        #print("extinct: {}".format(collections.Counter(df['extinct'])))
        #print("max(df['depth']): {}".format(max(df['depth'])))

        #
        if assign:
            self.df = df
        else:
            return df

        #print("All done parsing raw input.")


    def add_time(self, write=False, keep_text=False):
        '''
        gets organism age from wikipedia, but it's pretty slow
        optionally pass a filepath to `write` to write to csv every 100 iterations
        '''
        # load geological time data
        self.gt = pd.read_csv(GEO_TIME)

        # create dataframe for free text
        if keep_text:
            box_df = self.df[['id', 'name']].copy()
            box_df['text'] = None
        
        # fill time each organism
        self.df['Begin'] = None
        self.df['End'] = None
        for i, row in self.df.iterrows():
            #d, box = get_dates(row['name'], self.gt)
            d, cf_text = get_dates_and_cf(row['name'], self.gt)

            self.df.at[i,'Begin'] = d[0]
            self.df.at[i,'End'] = d[1]

            if keep_text:
                box_df.at[i, 'text'] = cf_text

            if i % 100 == 0:
                print(i, end='', flush=True)
                if write:
                    self.df.to_csv(write, index=False)
                    if keep_text:
                        box_df.to_csv(write.replace(".csv", "-text.csv"), index=False)
            elif i % 10 == 0:
                print('.', end='', flush=True)

        # write out at the end
        if write:
            self.df.to_csv(write, index=False)
            if keep_text:
                box_df.to_csv(write.replace(".csv", "-text.csv"), index=False)

    def return_data(self):
        '''returns the data as df'''
        if self.df is None:
            print("Must load data first. Try load_raw_file() or fetch_tol_data() and then parse_raw()")
        else:
            return self.df

    def write_prep_data(self, df_file, links_file):
        '''writes the prepared nodes df and links list to files'''
        if self.df is None: 
            print("Must load data first. Try load_raw_file() or fetch_tol_data() and then parse_raw()")
        else:
            # write
            self.df.to_csv(df_file, index=False)
            print("data written to {}".format(df_file))

    def load_prep_data(self, df_file, root):
        '''reads the prepared nodes df and links list
        must pass the id of the root node'''
        # read
        self.df = pd.read_csv(df_file)
        self.root = root
        self.root_age = self.df[self.df['id'] == self.root].squeeze()['Begin']

        # create links_dict
        links_dict = {}
        for i, this_row in self.df.iterrows():
            this_id = int(this_row['id'])
            # get parents
            if isinstance(this_row['ancestor'], int):
                parents = [this_row['ancestor']]
            else:
                parents = list(this_row['ancestor'])
                print(this_row)
            # add parents
            try:
                # add parents if the id already exists
                links_dict[this_id]['parents'] += parents
                # filter to only unique
                links_dict[this_id]['parents'] = list(set(links_dict[this_id]['parents']))
            except KeyError:
                # add this_id
                links_dict[this_id] = {}
                # add parent to this_id
                links_dict[this_id]['parents'] = parents
                links_dict[this_id]['children'] = []

            # add as child to parent node
            for parent in parents:
                try:
                    # add as child if parent node already exists
                    links_dict[parent]['children'].append(this_id)
                except KeyError:
                    # if not, create parent node and add as child
                    links_dict[parent] = {}
                    links_dict[parent]['children'] = [this_id]
                    links_dict[parent]['parents'] = []
                # filter to only unique
                links_dict[parent]['children'] = list(set(links_dict[parent]['children']))

        self.links_dict = links_dict

    def fix_ages(self):
        """fixes missing and weird ages
        starts at id passed as root and goes through all descendents. """
        
        this_gen = self.links_dict[self.root]['children']
        while len(this_gen) > 0:
            next_gen = []
            for c in this_gen:
                self.fix_this_age(c)
                #
                next_gen += self.links_dict[c]['children']
            #
            this_gen = next_gen

    def fix_this_age(self, c):

        this_row = self.df[self.df['id']==c].squeeze()
        this_age = this_row['Begin']
        parent = self.links_dict[c]['parents']
        parent_df = self.df[self.df['id'].isin(parent)]
        begin_age = float(parent_df['Begin'])

        # if there's an issue, fix it
        if (np.isnan(this_age)) \
           | (this_age > (begin_age * 0.95)) \
           | (this_age > (self.root_age * 0.95)):
            kids = self.links_dict[c]['children']
            if len(kids) > 0:
                kids_df = self.df[self.df['id'].isin(kids)]

                kids_max = np.nanmax(np.array(kids_df['Begin']))
                if (np.isnan(kids_max)) | (kids_max >= begin_age):
                    end_age = float(parent_df['End'])
                else:
                    end_age = kids_max
            else:
                end_age = float(parent_df['End'])
            # assign results
            if (this_age > (begin_age * 0.95)) & ~(this_age > (self.root_age * 0.95)):
                new_begin = np.round(np.max([(begin_age * 0.85), np.mean([begin_age, end_age])]), 1)
                self.df.at[self.df['id']==c, 'Begin'] = new_begin
                self.df.at[self.df['id']==c, 'End'] = np.min([new_begin, float(parent_df['End']), this_row['End']])
            else:
                new_begin = np.round(np.mean([begin_age, end_age]), 1)
                self.df.at[self.df['id']==c, 'Begin'] = new_begin
                self.df.at[self.df['id']==c, 'End'] = np.min([new_begin, float(parent_df['End'])])

    def load_text_data(self, text_file, cf_text=True):
        """"""
        self.text_df = pd.read_csv(text_file)
        try:
            self.text_df = self.text_df[self.text_df['id']\
                                    .isin(list(self.df['id']))]
        except AttributeError:
            print("FAILURE: self.df doesn't exist. Load some data first.")
            return None

        # filter to only classification words
        if cf_text:
            # define classification words (the next word after any of these will be kept)
            #self.CF_WORDS = ['kingdom', 'phylum', 'class', 'subclass', 'order', 'family', 'genus', 'species', 'clade']
            self.CF_WORDS = ['kingdom', 'phylum', 'class', 'subclass', 'order']
            # loop of all rows
            for i, row in self.text_df.iterrows():
                if type(row['text']) == str:
                    this_text = row['text'].replace("  ", " ").split(" ") # filter double spaces

                    # collect classification
                    this_cf = []
                    for j, t in enumerate(this_text):
                        if t in self.CF_WORDS:
                            this_cf.append(this_text[j+1])
                    this_cf = list(set(this_cf))
                    this_cf += row['name'].split(' ') # at species name

                    # assign to row in text_df
                    #self.text_df.at[i, 'text'] = ' '.join(this_cf)
                    if cf_text == 'add':
                        self.text_df.at[i, 'text'] = row['text'] + ' '.join(this_cf) + ' '.join(this_cf) + ' '.join(this_cf) + ' '.join(this_cf) 
                    else:
                        self.text_df.at[i, 'text'] = ' '.join(this_cf)

        # fill in missing text rows
        this_gen = self.links_dict[self.root]['children']
        while len(this_gen) > 0:
            next_gen = []
            for c in this_gen:
                self.fix_this_text(c)
                #
                next_gen += self.links_dict[c]['children']

            this_gen = next_gen

    def fix_this_text(self, c):
        # set up id to check
        check = c
        fix = True
        while fix == True:
            fix = False
            this_row = self.text_df[self.text_df['id']==check].squeeze()
            this_text = this_row['text']
            if type(this_text) != str:
                fix = True
            elif len(this_text) < 50:
                fix = True
            else:
                # add parents text for better downward flow
                try:
                    parent = self.links_dict[c]['parents'][0]
                    parent_row = self.text_df[self.text_df['id']==parent].squeeze()
                    this_text += ' '+parent_row['text'] 
                except:
                    pass
                #
                this_text += ' '+this_row['name']# add the original name
                self.text_df.at[self.text_df['id']==c, 'text'] = this_text

            if fix == True:
                check = self.links_dict[check]['parents'][0]
                

    def load_all_XY(self, mode='pca'):
        """mode must be 'pca' or 'tsne' or 'mds'
        """
        try:
            docs = list(self.text_df['text'])
        except AttributeError:
            print("FAILURE: self.text_df doesn't exist. Run PhyloGraph.load_text_data() first.")
            return None
        #
        vectorizer = CountVectorizer(max_df=0.5, min_df=0)
        vectors = vectorizer.fit_transform(docs)
        print("vocab shape: {}".format(vectors.shape))
        X = vectors.toarray()

        if mode == 'pca':
            X_pca = PCA(n_components=2).fit(X).transform(X)
            XY = pd.DataFrame(X_pca, columns = ['x', 'y'])
            print("loaded PCA data")
        else:
            X_pca = PCA(n_components=50).fit(X).transform(X)
            if mode == 'tsne':
                X_tsne = TSNE().fit(X_pca)
                XY = pd.DataFrame(X_tsne.embedding_, columns=['x', 'y'])
                print("loaded t-sne data")
            elif mode == 'mds':
                dists = pdist(X_pca, cosine)
                dist_matrix = pd.DataFrame(squareform(dists), 
                                       columns=self.text_df['id'], 
                                       index=self.text_df['id'])

                scaler = MDS(dissimilarity='precomputed', random_state=123)
                XY = pd.DataFrame(scaler.fit_transform(dist_matrix))
                XY.rename(index=str, columns={0:'x', 1:'y'}, inplace=True)
                print("loaded MDS data")
            else:
                print("mode must be one of 'pca' or 'tsne' or 'mds'")
                return None
        #
        self.df['x'] = list(XY['x'])
        self.df['y'] = list(XY['y'])


    def load_XY(self, depth, mode='pca'):
        """mode must be 'pca' or 'tsne' or 'mds'
        depth is how many generations to go with clustering
        before switching to jitter
        """
        try:
            docs = list(self.text_df['text'])
        except AttributeError:
            print("FAILURE: self.text_df doesn't exist. Run PhyloGraph.load_text_data() first.")
            return None
        #
        vectorizer = CountVectorizer(max_df=0.5, min_df=0)
        vectors = vectorizer.fit_transform(docs)
        print("vocab shape: {}".format(vectors.shape))
        X = vectors.toarray()

        if mode == 'pca':
            X_pca = PCA(n_components=2).fit(X).transform(X)
            XY = pd.DataFrame(X_pca, columns = ['x', 'y'])
            print("loaded PCA data")
        else:
            X_pca = PCA(n_components=50).fit(X).transform(X)
            if mode == 'tsne':
                X_tsne = TSNE().fit(X_pca)
                XY = pd.DataFrame(X_tsne.embedding_, columns=['x', 'y'])
                print("loaded t-sne data")
            elif mode == 'mds':
                dists = pdist(X_pca, cosine)
                dist_matrix = pd.DataFrame(squareform(dists), 
                                       columns=self.text_df['id'], 
                                       index=self.text_df['id'])

                scaler = MDS(dissimilarity='precomputed', random_state=123)
                XY = pd.DataFrame(scaler.fit_transform(dist_matrix))
                XY.rename(index=str, columns={0:'x', 1:'y'}, inplace=True)
                print("loaded MDS data")
            else:
                print("mode must be one of 'pca' or 'tsne' or 'mds'")
                return None
        #
        self.df['x'] = list(XY['x'])
        self.df['y'] = list(XY['y'])

    def rejitter_XY(self):
        """
        you have to call this before create_plot_data
        or it doesnt work right."""
        #depth_mult = np.sqrt(node['Begin'])
        #nodes_list[i]['x'] = parent[0]['x'] + (rn() * depth_mult)
        #nodes_list[i]['y'] = parent[0]['y'] + (rn() * depth_mult)

        #
        this_gen = self.links_dict[self.root]['children']
        while len(this_gen) > 0:
            next_gen = []
            for c in this_gen:
                # get parent
                parent = self.links_dict[c]['parents'][0]
                parent_row = self.df[self.df['id'] == parent].squeeze()
                this_row = self.df[self.df['id'] == c].squeeze()
                # pass down x and y plus the jitter
                self.df.at[self.df['id']==c, 'x'] = parent_row['x']+ (rn() * this_row['log_time'] / 4)#* np.sqrt(this_row['Begin']))
                self.df.at[self.df['id']==c, 'y'] = parent_row['y']+ (rn() * this_row['log_time'] / 4)#* np.sqrt(this_row['Begin']))
                # get next generation
                next_gen += self.links_dict[c]['children']
            #
            this_gen = next_gen


class phyloGraph():
    '''
    takes in an object of class phyloData
    builds a plot and optionally publishes it to plot.ly
    '''
    def __init__(self, phyloData, username=None, api_key=None):
        print("the plotting of phylo")
        # load df and links_dict
        self.df = phyloData.df
        self.links_dict = phyloData.links_dict

        # connect to plot.ly
        if (username is not None) & (api_key is not None):
            plotly.tools.set_credentials_file(username=username, api_key=api_key)
        elif (username is None) & (api_key is None):
            pass
        else:
            print("must pass BOTH username & api_key in order to publish.")

    def search_name(self, search_name):
        print(self.df[self.df['name'].str.contains(search_name)][['name', 'id']])

    def get_descendants(self, pick, mode, start=time()):
        '''get all descendants from an id
        mode can be either:
            'filter' - filters self.df to all descendants of pick
            'focus' - creates 'kin' column and assigns 1 to descendants and parents and 0 to all others
        '''
        #print("AAA {} getting descendents".format(np.round(time()-start, 1)))
        # get descendants
        kids = []
        this_gen = self.links_dict[pick]['children']
        depth = 0
        while len(this_gen) > 0:
            kids += this_gen
            depth += 1
            next_gen = []
            for c in this_gen:
                next_gen += self.links_dict[c]['children']
            if depth >= self.max_depth:
                break
            else:
                this_gen = next_gen
        # 
        keepers = [pick]
        keepers += kids
        #print("{} keepers at depth {} ".format(len(keepers), depth))
        #
        if mode == 'filter':
            #print("AAA {} filtering plot_df to {} keepers ".format(np.round(time()-start, 1), len(keepers)))
            self.plot_df = self.df[self.df['id'].isin(keepers)]
            self.plot_df.reset_index(inplace=True, drop=True)

            # add log time
            self.plot_df['log_time'] = np.log1p(self.plot_df['Begin'])

        elif mode == 'focus':
            # add ancestors
            parent = self.links_dict[pick]['parents']
            # add parent's other kids
            #keepers += self.links_dict[parent[0]]['children']
            
            while parent in self.plot_df['id'].values:
            #while len(parent) > 0:
                keepers += parent

                # record this one's kids
                siblings = self.links_dict[parent[0]]['children']

                # check the next one
                parent = self.links_dict[parent[0]]['parents']

            # add kids of root node
            keepers += siblings

            # add kin column
            self.plot_df['kin'] = 0
            self.plot_df['kin'][self.plot_df['id'].isin(keepers)] = 1
            # return list of kin
            return keepers
        else:
            print("FAILURE: get_descendants(mode) only valid options are 'filter' and 'focus'")

    def create_plot_df(self, root, max_depth = 50):
        start = time()
        # subset to only children of the root
        self.root = root
        self.root_age = self.df[self.df['id'] == self.root].squeeze()['Begin']
        self.max_depth = max_depth

        #print("AAA {} root : {} ({} MYA)".format(np.round(time()-start, 1), self.root, self.root_age))
        self.get_descendants(self.root, mode='filter', start=start)

        # filtering out the ones with Begin == 0
        #print("AAA {} filtering out the ones with Begin == 0".format(np.round(time()-start, 1)))
        self.plot_df = self.plot_df.loc[self.plot_df['Begin'] > 0.1]
        #print("AAA {} reseting index".format(np.round(time()-start, 1)))
        self.plot_df.reset_index(inplace=True, drop=True)


    def create_plot_data(self, 
                         color_attr = 'extinct',
                         Z_dim = 'log_time',
                         Z_dim_mult = 1, # set to -1 if using 'depth' for Z_dim
                         max_nodes = 50000):
        ''''''
        # subset to max_nodes and max_depth
        #df = self.df.head(max_nodes)
        #df = df.reset_index(inplace=True, drop=True)
        ### PUT THIS ^ IN get_descendants()

        # assign args
        self.color_attr = color_attr
        self.Z_dim = Z_dim
        self.Z_dim_mult = Z_dim_mult

        ## create links list
        links_list = []
        for i, n in self.plot_df.iterrows():
            links_list.append({
                'source': n['id'], 
                'target': n['ancestor'], 
                'value': n['num_kids']
            })

        L=len(links_list)

        # make list of tuples for edges
        Edges=[(links_list[k]['source'], links_list[k]['target']) for k in range(L)]

        # create layout
        labels=[]
        group=[]
        layt = []
        for i, node in self.plot_df.iterrows():
            labels.append("{} -- {} ({} MYA)".format(node['name'], node['id'], node['Begin']))
            # create color key
            group.append(node[self.color_attr])
            # create layout list
            d = node[self.Z_dim]
            layt.append([node['x'], 
                         node['y'], 
                         #(d*self.Z_dim_mult)+np.random.uniform(-0.1,0.1,1)[0]])    
                         d*self.Z_dim_mult])

        # make nodes
        Xn=[layt[k][0] for k in range(len(layt))]# x-coordinates of nodes
        Yn=[layt[k][1] for k in range(len(layt))]# y-coordinates
        Zn=[layt[k][2] for k in range(len(layt))]# z-coordinates

        # make edges
        Xe=[]
        Ye=[]
        Ze=[]

        for e in Edges:
            try: 
                e0 = self.plot_df[self.plot_df['id'] == e[0]].index.values[0]
            except:
                e0 = 0
            try:
                e1 = self.plot_df[self.plot_df['id'] == e[1]].index.values[0]
            except:
                e1 = 0
            #
            Xe+=[layt[e0][0],layt[e1][0], None]# x-coordinates of edge ends
            Ye+=[layt[e0][1],layt[e1][1], None]  
            Ze+=[layt[e0][2],layt[e1][2], None] 

        # make traces
        this_text = self.plot_df.loc[0, 'name']
        ### this_text = self.plot_df.loc[self.plot_df['id']==pick]['name'].values[0]

        # time line
        root_row = self.plot_df[self.plot_df['id'] == self.root].squeeze()
        trace0 = go.Scatter3d(
                x=[root_row['x'], root_row['x'], None],
                y=[root_row['y'], root_row['y'], None],
                z=[np.max(Zn) + 0.25, np.min(Zn), None],
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=5),
                hoverinfo='none'
            )
        trace0t = go.Scatter3d(
                x=[root_row['x'], root_row['x'], root_row['x'],],
                y=[root_row['y'], root_row['y'], root_row['y']],
                z=[np.log1p(1), np.log1p(5), np.log1p(20), np.log1p(50), np.log1p(150)],
                mode='text',
                text=['1 MYA', '5 MYA', '20 MYA', '50 MYA', '150 MYA']
            )

        # lines
        trace1=go.Scatter3d(x=Xe,
                       y=Ye,
                       z=Ze,
                       mode='lines',
                       #opacity=0.65,
                       opacity=0.33,
                       line=dict(color='rgb(125,125,125)', width=1),
                       hoverinfo='none'
                       )

        # nodes
        trace2=go.Scatter3d(x=Xn,
                       y=Yn,
                       z=Zn,
                       mode='markers',
                       name='actors',
                       marker=dict(symbol='circle',
                                     size=6,
                                     color=group,
                                     #opacity=0.55,
                                     opacity=0.33,
                                     colorscale='Viridis'
                                     ),
                       text=labels,
                       hoverinfo='text',
                       textfont=dict(
                                family='sans serif',
                                size=18,
                                color='#ff7f0e'
                            )
                       )

        axis=dict(showbackground=False,
                  showline=False,
                  zeroline=False,
                  showgrid=False,
                  showticklabels=False,
                  title=''
                  )

        layout = go.Layout(
                 #title=this_title,
                 title=this_text,
                 width=1000,
                 height=1000,
                 showlegend=False,
                 scene=dict(
                     xaxis=dict(axis),
                     yaxis=dict(axis),
                     zaxis=dict(axis),
                ),
             margin=dict(
                t=100
            ),
            hovermode='closest',
            #dragmode='turntable', # https://plot.ly/python/reference/#layout-dragmode
            annotations=[
                dict(
                   showarrow=False,
                    text= "Root node: {}".format(this_text),
                    xref='paper',
                    yref='paper',
                    x=0,
                    y=0.1,
                    xanchor='left',
                    yanchor='bottom',
                    font=dict(
                    size=14
                    )
                    )
                ]   
            )

        # assign traces
        self.layout = layout
        self.plot_data=[trace0, trace0t, trace1, trace2]
        self.focus_plot_data = None

        #
        print("Loaded plot data. Root node: {}".format(self.root))


    def focus_plot(self, focus, add_links = True):
        '''makes the "focus" nodes
        highlights the kinfolk of the id you pass to focus arg'''
        # get kinfolk of focus node
        kin = self.get_descendants(focus, mode='focus')
        self.focus_df = self.plot_df[self.plot_df['kin']==1]
        self.focus_df.reset_index(inplace=True, drop=True)

        focus_row = self.focus_df.loc[self.focus_df['id']==focus].squeeze()

        ## create links list
        links_list = []
        for i, n in self.focus_df.iterrows():
            links_list.append({
                'source': n['id'], 
                'target': n['ancestor'], 
                'value': n['num_kids']
            })

        L=len(links_list)

        #
        Edges=[(links_list[k]['source'], links_list[k]['target']) for k in range(L)]

        # 
        labels=[]
        group=[]
        layt = []
        for i, node in self.focus_df.iterrows():
            # create text labels
            if add_links:
                try: # why this try/except?
                    #this_page_id = wikipedia.search(node['name'], results=1)[0]
                    labels.append('<a href="https://en.wikipedia.org/wiki/{}">{} -- {} ({} MYA)</a>'.format(node['name'], node['name'], node['id'], node['Begin']))
                except:
                    labels.append("{} -- {} ({} MYA)".format(node['name'], node['id'], node['Begin']))
            else:
                labels.append("{} -- {} ({} MYA)".format(node['name'], node['id'], node['Begin']))

            # create color key
            group.append(node[self.color_attr])
            # create layout list
            d = node[self.Z_dim]

            layt.append([node['x'], 
                         node['y'], 
                         #(d*self.Z_dim_mult)+np.random.uniform(-0.1,0.1,1)[0]])    
                         d*self.Z_dim_mult])    

        # semi-hack to always make sure there are two colors
        if self.color_attr == 'extinct':
            group.append(0)
            group.append(2)
        # because if it's all one or the other the color changes
        # and for some reason there are no 1's

        # make nodes
        Xn=[layt[k][0] for k in range(len(layt))]# x-coordinates of nodes
        Yn=[layt[k][1] for k in range(len(layt))]# y-coordinates
        Zn=[layt[k][2] for k in range(len(layt))]# z-coordinates

        # make edges
        Xe=[]
        Ye=[]
        Ze=[]
        for e in Edges:
            try: 
                e0 = self.focus_df[self.focus_df['id'] == e[0]].index.values[0]
            except:
                e0 = 0
            try:
                e1 = self.focus_df[self.focus_df['id'] == e[1]].index.values[0]
            except:
                e1 = 0
            #
            Xe+=[layt[e0][0],layt[e1][0], None]# x-coordinates of edge ends
            Ye+=[layt[e0][1],layt[e1][1], None]  
            Ze+=[layt[e0][2],layt[e1][2], None] 

        # make traces
        #this_text = self.focus_df.loc[self.focus_df['id']==focus]['name'].values[0]
        this_title = '<a href="https://en.wikipedia.org/wiki/{name}">{name}</a> ({year} MYA)'\
                                        .format(name = focus_row['name'],
                                                year = focus_row['Begin'])

        # kinfolk lines
        trace1k=go.Scatter3d(x=Xe,
                       y=Ye,
                       z=Ze,
                       mode='lines',
                       opacity=1,
                       line=dict(color='rgb(125,125,125)', width=3),
                       hoverinfo='none'
                       )

        # kinfolk nodes
        trace2k=go.Scatter3d(x=Xn,
                       y=Yn,
                       z=Zn,
                       mode='markers',
                       name='actors',
                       marker=dict(symbol='circle',
                                     size=10,
                                     color=group,
                                     colorscale='Viridis',
                                     line=dict(color='rgb(50,50,50)', width=0.5)
                                     ),
                       text=labels,
                       hoverinfo='text'
                       )

                # kinfolk nodes
        tracef=go.Scatter3d(x=[focus_row['x']],
                       y=[focus_row['y']],
                       z=[focus_row[self.Z_dim]*self.Z_dim_mult],
                       mode='markers',
                       name='actors',
                       marker=dict(symbol='circle',
                                     size=20,
                                     opacity = 0.6,
                                     color=[1],
                                     colorscale='Viridis'),
                       text=labels,
                       hoverinfo='text'
                       )

        self.focus_plot_data = [trace1k, trace2k, tracef]
        self.layout.title = this_title

        #
        print("Loaded plot data. Highlighting: {}".format(focus))

    def render_plot(self, publish=False, filename="testplot"):
        if self.focus_plot_data:
            plot_data = self.plot_data + self.focus_plot_data
        else:
            plot_data = self.plot_data

        #fig=go.Figure(data=plot_data, layout=self.layout)
        ###########
        fig=go.FigureWidget(data=plot_data, layout=self.layout)

        ## scatter = fig.data[3]
        ## 
        ## # create our callback function
        ## def update_point(trace, points, selector):
        ##     c = list(scatter.marker.color)
        ##     s = list(scatter.marker.size)
        ##     for i in points.point_inds:
        ##         print(i)
        ##         c[i] = '#bae2be'
        ##         s[i] = 20
        ##         scatter.marker.color = c
        ##         scatter.marker.size = s
        ## 
        ## scatter.on_click(update_point)
        ## ##########

        if publish:
            self.plot = py.iplot(fig, filename=filename)
        else:
            iplot(fig)

    def unfocus(self):
        self.focus_plot_data = None

    def refocus(self, focus):
        self.focus_plot(focus)
        self.render_plot()

    def open_plot(self):
        webbrowser.open(self.plot.resource, new=2)

