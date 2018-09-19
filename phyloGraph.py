import requests
import collections
import json
import pandas as pd
import numpy as np
import re

from sklearn.decomposition import PCA

import webbrowser
import wikipedia

import plotly
import plotly.plotly as py
from plotly.offline import iplot
import plotly.graph_objs as go

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


def get_dates(this_query, gt):
    """gets the earliest and latest date for an organism
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
        fr = box.split("fossil_range")[1].split('\n')[0]
        useful = re.sub("[^A-Za-z0-9\.]+", " ", fr)
        useful = useful.replace("Late ","Late_").replace("Middle ","Middle_").replace("Early ", "Early_").split(" ")
        
        # try to get actual MYA times
        times = [trynum(n) for n in useful]
        times = [n for n in times if n is not None]
        if len(times) >= 2:
            return (max(times), min(times))
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
            return (max(dates), min(dates))
    except:
        return (None, None)



class phyloData:
    '''
    either fetches data from the Tree of Life website,
    loads raw data from file,
    or loads prepared data from a files.
    '''
    def __init__(self):
        print("the data of phylo")
        self.df = None
        self.links_list = None
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
        df, links_list = self.parse_raw(raw_search, named=False, assign=False)
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
        # parse to df and links_list
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
        df and links_list instead of assigning them to self'''
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

        # create links list
        links_list = []
        for n in nodes_list:
            links_list.append({
                'source': n['id'], 
                'target': n['ancestor'], 
                'value': n['num_kids']
            })

        #print("len(links_list): {}".format(len(links_list)))
        #print("first 3 links: {}".format(links_list[:3]))

        # pred df
        df = pd.DataFrame(nodes_list)
        #print("df.head():\n{}".format(df.head()))
        #print("phylesis: {}".format(collections.Counter(df['phylesis'])))
        #print("extinct: {}".format(collections.Counter(df['extinct'])))
        #print("max(df['depth']): {}".format(max(df['depth'])))

        #
        if assign:
            self.df = df
            self.links_list = links_list
        else:
            return df, links_list

        #print("All done parsing raw input.")

    def write_prep_data(self, df_file, links_file):
        '''writes the prepared nodes df and links list to files'''
        if (self.df is None) | (self.links_list is None): 
            print("Must load data first. Try load_raw_file() or fetch_tol_data() and then parse_raw()")
        else:
            # write
            self.df.to_csv(df_file, index=False)
            with open(links_file, 'w') as f:
                f.write('\n'.join([json.dumps(l) for l in self.links_list]))
            print("data written to {} and {}".format(df_file, links_file))

    def load_prep_data(self, df_file, links_file):
        '''reads the prepared nodes df and links list'''
        # read
        self.df = pd.read_csv(df_file)
        with open(links_file, 'r') as f:
            lines = f.read().splitlines()
            self.links_list = [json.loads(l) for l in lines]

    def add_time(self):
        # load geological time data
        self.gt = pd.read_csv(GEO_TIME)
        
        # fill time each organism
        self.df['Begin'] = None
        self.df['End'] = None
        for i, row in self.df.iterrows():
            d = get_dates(row['name'], self.gt)
            # if nothing, look up ancestor
            #if not all(d):
            #    a = (int(self.df.loc[self.df['id']==row['ancestor']]['Begin']), \
            #        int(self.df.loc[self.df['id']==row['ancestor']]['End']))
            #    if all(a):
            #        d = (np.mean(a), a[1])
            #    else:
            #        d = (None, None)
            #    ##### THIS NEEDS TO BE INSIDE A TRY, MAYBE NEED TO MOVE get_names() INTO THE MODULE AND PUT THIS INSIDE IT

            self.df.at[i,'Begin'] = d[0]
            self.df.at[i,'End'] = d[1]

            if i % 100 == 0:
                print(i, end='', flush=True)
            elif i % 10 == 0:
                print('.', end='', flush=True)


    def return_data(self):
        '''returns the data as df and links_list'''
        if (self.df is None) | (self.links_list is None):
            print("Must load data first. Try load_raw_file() or fetch_tol_data() and then parse_raw()")
        else:
            return self.df, self.links_list

class phyloGraph():
    '''
    takes in a dataframe of nodes and a list of links
    builds a plot and optionally publishes it to plot.ly
    '''
    def __init__(self, df, links_list, username=None, api_key=None):
        print("the plotting of phylo")
        self.df = df
        self.links_list = links_list
        # 
        if (username is not None) & (api_key is not None):
            plotly.tools.set_credentials_file(username=username, api_key=api_key)
        elif (username is None) & (api_key is None):
            pass
        else:
            print("must pass BOTH username & api_key in order to publish.")

    def search_name(self, search_name):
        print(self.df[self.df['name'].str.contains(search_name)][['name', 'id']])

    def create_plot_data(self, 
                         highlight = 'all',
                         max_nodes = 5000,
                         max_depth = 20,
                         color_attr = 'extinct',
                         Z_dim = 'depth'):
        ''''''
        if highlight == 'all':
            pick = self.df.ix[0, 'id']
        else:
            pick = highlight

        # subset to max_nodes and max_depth
        df = self.df[self.df['depth'] <= max_depth].head(max_nodes)
        df = df.reset_index()

        # get kinfolk
        ma = int(df['ancestor'][df['id'] == pick])
        #print("ma: {}".format(ma))
        ancestors = []

        this_id = pick
        for i in range(max_depth):
            this_row = df[df['id'] == this_id]
            if this_row.shape[0] == 0:
                break
            else:
                this_ancestor = int(this_row['ancestor'])
                ancestors.append(this_ancestor)
                this_id = this_ancestor
            
        #print("{} ancestors".format(ancestors))

        kin = [pick]

        # add descendants
        for i, row in df.iterrows():
            if int(row['ancestor']) in kin:
                kin.append(int(row['id']))

        # add siblings
        for i, row in df.iterrows():
            if int(row['ancestor']) == ma:
                kin.append(int(row['id']))

        # add ancestors
        kin += ancestors
        #print(len(kin))

        # subset
        df['kin'] = 0
        df['kin'][df['id'].isin(kin)] = 1

        links_list = [l for l in self.links_list if l['target'] in list(df['id'])]
        L=len(links_list)
        #print("len(links_list): {}".format(L))

        #
        Edges=[(links_list[k]['source'], links_list[k]['target']) for k in range(L)]
        #print("Edges {}".format(Edges[:5]))

        # 
        labels=[]
        group=[]
        alpha = []
        layt = []
        for i, node in df.iterrows():
            # create text labels
            labels.append(str(node['depth']) + ' ' + node['name'])
            # create color key
            group.append(node[color_attr])
            # create opacity key
            alpha.append(node['kin'])
            # create layout list
            d = node[Z_dim]
            # # PCA
            # # layt.append(list(X_pca[i]) + [d * -1])
            layt.append([node['x'], 
                         node['y'], 
                         (d*-1)+np.random.uniform(-0.1,0.1,1)[0]])    
            

        #print(labels[:3])
        #print(group[:3])
        #print(alpha[:3])
        #print(len(layt))
        #print(layt[:3])

        # make nodes
        Xn=[layt[k][0] for k in range(len(layt))]# x-coordinates of nodes
        Yn=[layt[k][1] for k in range(len(layt))]# y-coordinates
        Zn=[layt[k][2] for k in range(len(layt))]# z-coordinates

        # check if they're kinfolk
        Xnk=[layt[k][0] for k in range(len(layt)) if alpha[k]==1]# x-coordinates of nodes
        Ynk=[layt[k][1] for k in range(len(layt)) if alpha[k]==1]# y-coordinates
        Znk=[layt[k][2] for k in range(len(layt)) if alpha[k]==1]# z-coordinates
        groupk=[group[k] for k in range(len(layt)) if alpha[k]==1]# color
        #print(len(Xnk))

        # make edges
        Xe=[]
        Ye=[]
        Ze=[]
        Xek=[]
        Yek=[]
        Zek=[]
        for e in Edges:
            try: 
                e0 = df[df['id'] == e[0]].index.values[0]
            except:
                e0 = 0
            try:
                e1 = df[df['id'] == e[1]].index.values[0]
            except:
                e1 = 0
            #
            Xe+=[layt[e0][0],layt[e1][0], None]# x-coordinates of edge ends
            Ye+=[layt[e0][1],layt[e1][1], None]  
            Ze+=[layt[e0][2],layt[e1][2], None] 
            # check if they're kinfolk
            if e[0] in kin:
                Xek+=[layt[e0][0],layt[e1][0], None]
                Yek+=[layt[e0][1],layt[e1][1], None]
                Zek+=[layt[e0][2],layt[e1][2], None]

        #print(len(Ze))
        #print(len(Zek))

        # make traces
        this_title = df.loc[0, 'name']
        this_text = df.loc[df['id']==pick]['name'].values[0]

        # lines
        trace1=go.Scatter3d(x=Xe,
                       y=Ye,
                       z=Ze,
                       mode='lines',
                       opacity=0.7,
                       line=dict(color='rgb(125,125,125)', width=1),
                       hoverinfo='none'
                       )
        # kinfolk lines
        trace1k=go.Scatter3d(x=Xek,
                       y=Yek,
                       z=Zek,
                       mode='lines',
                       opacity=1,
                       line=dict(color='rgb(125,125,125)', width=1.5),
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
                                     opacity=0.6,
                                     colorscale='Viridis'
                                     ),
                       hoverinfo='skip'
                       )

        # kinfolk nodes
        trace2k=go.Scatter3d(x=Xnk,
                       y=Ynk,
                       z=Znk,
                       mode='markers',
                       name='actors',
                       marker=dict(symbol='circle',
                                     size=8,
                                     color=groupk,
                                     colorscale='Viridis',
                                     line=dict(color='rgb(50,50,50)', width=0.5)
                                     ),
                       text=labels,
                       hoverinfo='text'
                       )

        axis=dict(showbackground=False,
                  showline=False,
                  zeroline=False,
                  showgrid=False,
                  showticklabels=False,
                  title=''
                  )

        layout = go.Layout(
                 title=this_title,
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
            annotations=[
                   dict(
                   showarrow=False,
                    text=this_text,
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
                ],    )

        # assign traces
        self.layout = layout
        self.plot_data=[trace1, trace1k, trace2, trace2k]

        #
        print("Loaded plot data. Highlighting {}".format(pick))

    def render_plot(self, publish=False, filename="testplot"):
        fig=go.Figure(data=self.plot_data, layout=self.layout)

        if publish:
            self.plot = py.iplot(fig, filename=filename)
        else:
            iplot(fig)

    def open_plot(self):
        webbrowser.open(self.plot.resource, new=2)

