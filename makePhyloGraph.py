import phyloGraph as ph
import argparse

#vim ~/.Plotly/.Credentials

parser = argparse.ArgumentParser(description='''
Makes a plot of the evolutionary tree, starting at the node specified at --root.

You MUST have a plotly account with the username and api key in a file
named ~/.Plotly/.Credentials for this script to work.
You can go to http://plot.ly to sign up for an account. It's free.

You can run `python searchTreeOfLife.py {search name} to search for
a list of an organism and it's corresponding id.
''')

parser.add_argument('--mode', type=str, default='tol', choices = ['tol', 'raw_file', 'prep_file'], help='The mode to fetch data. "raw" fetches from Tree of Life website. The other two fetch from a local file.')
parser.add_argument('--focus', type=int, help='The integer id of the node to start the graph from.')
parser.add_argument('-p', '--plotname', type=str, default='testplot', help='The name for your plot when you publish it to plot.ly website. Will overwrite a plot with the same name.')
parser.add_argument('-r', '--root', type=int, help='If you chose "tol" for --mode, pass the integer id of the node to start the graph from.')
parser.add_argument('-f', '--filename', type=str, help='If you chose "raw_file" or "prep_file" for --mode, pass the filename of the file to load here.')
parser.add_argument('-l', '--link_file', type=str, help='If you chose "prep_file" for --mode, you also have to pass the filename with the links in it.')

args = parser.parse_args()

if __name__ == "__main__":
    # ## Load Data
    pgd = ph.phyloData()

    # Pick one of the methods below for getting data
    if args.mode == 'tol':
        pgd.fetch_tol_data(args.root)
    elif args.mode == 'raw_file':
        pgd.load_raw_file(args.filename)
    elif args.mode == 'prep_file':
        pgd.load_prep_data(args.filename, args.link_file)
    else:
        print("Mode must be one of ['tol', 'raw_file', 'prep_file']")

    # ### return data for plotting
    df, links_list = pgd.return_data()
    pgp = ph.phyloGraph(df, links_list)

    # ### create plot
    if args.focus:
        highlight = args.focus
    else:
        highlight = 'all'
    pgp.create_plot_data(highlight=highlight, max_depth=30, max_nodes=50000)
    
    try:
        pgp.render_plot(publish=True, filename=args.plotname)
        # open in browser
        pgp.open_plot()
    except:
        print('You MUST have a plotly account with the username and api key in a file named ~/.Plotly/.Credentials for this script to work. You can go to http://plot.ly to sign up for an account. It's free.')



