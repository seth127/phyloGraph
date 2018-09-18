import phyloGraph as ph
import argparse

parser = argparse.ArgumentParser(description='''
Search Tree of Life website for a name of an organism.
The search will return all names that match the search string
along with an integer id. 

Pass that id to makePhyloGraph.py one of the following arguments
--root :: to start the tree at with that organism
OR
--highlight :: to highlight the ancestors and descendants of that organism.
''')

parser.add_argument('search', help='The text string to search for organism name matches on Tree of Life. Can be a partial match within the name.')

args = parser.parse_args()

if __name__ == "__main__":
    pgd = ph.phyloData()

    resp = pgd.search_name(args.search)

    print(resp)
