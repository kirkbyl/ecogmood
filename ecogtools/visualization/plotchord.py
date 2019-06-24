import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import collections
import brewer2mpl as brewcm


from matplotlib.path import Path
from ecogtools.tools import utilities as utils


def plot_graphs(graphDict, subset=None, nodeLabels=None, title='', radius=10, noderadius=0.07, alpha=0.3, linewidth=None, linestyle=None, nodecolor=None, edgecolor=None, nodefill=None, fig=None, ax=None, colormap=cm.plasma, **kwargs):
    """
    Plots networkx graphs as a chord diagram
    
    - graphDict is a dictionary of networkx graphs, keys correspond to graph index (0, 1, 2, 3 ...). Graph is either instance of networkx class
        or a dictionary with keys 'pos' and 'neg' if graph contains both increasing and decreasing connections (eg for ICNs)
    - subset specifies the indices (keys) of which graphs to plot
    """
            
    if subset is None:
        subset = range(len(graphDict))
        if len(subset) > 25:
            print 'Plotting first 25'
            subset = range(25)

    if nodeLabels is not None:        
        if nodecolor is not None:
            overwriteColor_node = False
        else:
            overwriteColor_node = True
        if edgecolor is not None:
            overwriteColor_edge = False
        else:
            overwriteColor_edge = True
        
    if fig == None and ax == None:
        fig, ax, ax1D = utils.custom_subplots(len(subset))
    else:
        ax1D = np.ravel(ax) 
    
    for idx, X in enumerate(subset):
        
        graph = graphDict[X]

        if isinstance(graph, dict) and 'pos' in graph:
            for kk, key in enumerate(['pos', 'neg']):
                # Connections with increased connectivity: solid line
                if key == 'pos':
                    linestyle='-'                 
                # Connections with decreased connectivity: dashed line
                elif key == 'neg':
                    #linestyle='--'
                    linestyle='-'
                
                if nodeLabels is not None:
                    if overwriteColor_node == True:
                        nodecolor = get_node_colors(nodeLabels, colormap=colormap)
                    if overwriteColor_edge == True:                    
                        edgecolor1, edgecolor2 = get_edge_colors(graph[key].edges(), nodeLabels, colormap=colormap)
                
                for edgecolor in [edgecolor1, edgecolor2]:
                    plot_graph_single(graph[key], fig=fig, ax=ax1D[idx], radius=radius, nodecolor=nodecolor, edgecolor=edgecolor, noderadius=noderadius, nodefill=nodefill, linewidth=linewidth, linestyle=linestyle, alpha=alpha)           
                ax1D[idx].set_title('ICN {0}'.format(str(X+1)), fontsize=11)
        else:
            
            if nodeLabels is not None:
                if overwriteColor_node == True:
                    nodecolor = get_node_colors(nodeLabels, colormap=colormap)
                if overwriteColor_edge == True:                    
                    edgecolor1, edgecolor2 = get_edge_colors(graph.edges(), nodeLabels, colormap=colormap)
                         
            for edgecolor in [edgecolor1, edgecolor2]:      
                plot_graph_single(graph, fig=fig, ax=ax1D[idx], radius=radius, nodecolor=nodecolor, edgecolor=edgecolor, noderadius=noderadius, nodefill=nodefill, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
            ax1D[idx].set_title('ICN {0}'.format(str(X+1)), fontsize=11)
            
    plt.suptitle(title, fontsize=14)
    plt.show()
				
    return fig
    
    
def plot_graph_single(graph, radius=10, ax=None, fig=None, nodecolor=None, edgecolor=None, noderadius=None, nodefill=None, linewidth=None, linestyle=None, alpha=None):
    """
    Plot single graph.
    """
    
    nodes = graph.nodes()
    edges = graph.edges()
    
    circ = CircosPlot(nodes, edges, radius, noderadius=noderadius, nodecolor=nodecolor, nodefill=nodefill, edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle, alpha=alpha, fig=fig, ax=ax)
    
    if len(circ.edges) >= 2:
        circ.draw()
    else:
        circ.add_nodes()
        
    return



def get_node_labels(abbrs, regDict):
    """
    Get common labels (eg. STC, AM, HPC etc) for patient-specific electrodes
    
    - abbrs: list of brain regions ['STC', 'AM', 'HP'] etc
    - regDict: dictionary of brain regions: {reg abbr: [electrodes/nodes]}
    """
    
    nodeLabels = collections.OrderedDict()
    
    for ii, rr in enumerate(abbrs):
        nodeLabels[rr] = []
        
        for reg, elecs in regDict.items():
            if rr in reg:
                nodeLabels[rr] = nodeLabels[rr] + elecs
    
    return nodeLabels
    
    
def graph_cmap(nColors, colormap=cm.plasma):
    """
    Color pallet for nodes and edges in cord diagram
    
    Inputs:
        - nColors: int, number of desired colors
        - colormap: matplotlib.cm colormap object (default: cm.plasma)
    """
    
    if hasattr(colormap, 'N'):
        # Exclude yellow green and pink from Set3
        if colormap.name == 'Set3':
            goodInds = [0,2,3,4,5,8,9,10,11]
            if nColors <= len(goodInds):
                cmap = [colormap.colors[cc] for cc in goodInds[0:nColors]]
        else:
            if colormap.N < 256 and nColors <= colormap.N:
    
                cmap = colormap.colors[0:nColors]
            else:
                cmap_all = colormap(np.linspace(0.1, 0.9, nColors))
                
                # Shuffle so similar colors are not right next to each other:
                cmap = np.vstack((cmap_all[::2], cmap_all[1::2]))
    else:
        print('Colormap not recognized')

    return cmap
    
    
    
def get_node_colors(nodeLabels, colormap=cm.plasma):
    """
    Define colors for nodes in graph
    
    - nodeLabels: dictionary of {label: [nodes]} (output of get_node_labels)
    """
    
    nColors = len(nodeLabels)
    colors = graph_cmap(nColors, colormap=colormap)
    
    nodecolor = []
    
    for ee, elecs in enumerate(nodeLabels.values()):
        
        for elec in elecs:
            
            nodecolor.append(colors[ee])

    return nodecolor
    
    
    
def get_node_fills(nodeLabels, emptyNodes=None):
    """
    Define whether to fill or unfill nodes -- unfilled nodes are those that do not have electrodes in that location
    
    - nodeLabels: dictionary of {label: [nodes]} (output of get_node_labels)
    """
    
    nodefill = []
    
    if emptyNodes is None:
        emptyNodes = []
    
    for ee, elecs in enumerate(nodeLabels.values()):
        
        for elec in elecs:
            
            if elec in emptyNodes:
                nodefill.append(False)
            else:
                nodefill.append(True)
    
    return nodefill



def get_edge_colors(edges, nodeLabels, colormap=cm.plasma):
    """
    Define colors for edges in graph
    
    - nodeLabels: dictionary of {label: [nodes]} (output of get_node_labels)
    - edges: list of edge tuples    
    """
    
    nColors = len(nodeLabels)
    colors = graph_cmap(nColors, colormap=colormap)
            
    edgecolor1 = []
    edgecolor2 = []

    for edge in edges:
        for ee, elecs in enumerate(nodeLabels.values()):
            if edge[0] in elecs:
                edgecolor1.append(colors[ee])
            if edge[1] in elecs:
                edgecolor2.append(colors[ee])
                            
    return edgecolor1, edgecolor2



class CircosPlot(object):
    def __init__(self, nodes, edges, radius, noderadius=None,
                 nodecolor=None, edgecolor=None,
                 nodeprops=None, edgeprops=None,
                 linewidth=None, linestyle=None,
                 nodefill=None, alpha=None,
                 figsize=(8, 8), ax=None, fig=None):
        self.nodes = nodes  # list of nodes
        self.edges = edges  # list of edge tuples

        # Make sure props are dictionaries if passed in
        # Node props
        if nodeprops is not None:
            if isinstance(nodeprops, dict):
                self.nodeprops = nodeprops
            else:
                raise TypeError("nodeprops must be a dictionary")
        else:
            self.nodeprops = {}
        # Edge props
        if edgeprops is not None:
            if isinstance(edgeprops, dict):
                self.edgeprops = edgeprops
            else:
                raise TypeError("edgeprops must be a dictionary")
        else:
            self.edgeprops = {}

        # Set colors. Priority: nodecolor > nodeprops > default
        # Node color
        if nodecolor is not None:
            self.nodecolor = nodecolor
        elif nodeprops:
            try:
                self.nodecolor = nodeprops.pop('facecolor')
            except KeyError:
                self.nodecolor = 'blue'
        else:
            self.nodecolor = 'blue'
        # Edge color
        if edgecolor is not None:
            self.edgecolor = edgecolor
        elif edgeprops:
            try:
                self.edgecolor = edgeprops.pop('edgecolor')
            except KeyError:
                self.edgecolor = 'black'
        else:
            self.edgecolor = 'black'
        
        if linewidth is None:
            self.lw = 1
        else:
            self.lw = linewidth

        if linestyle is None:
            self.ls = '-'
        else:
            self.ls = linestyle
            
        if noderadius is None:
            noderadius = 0.05
        
        if alpha is None:
            alpha = 1
        
        if nodefill is not None:
            self.nodefill = nodefill
        
        self.alpha = alpha
        self.radius = radius
        if fig is None:
            self.fig = plt.figure(figsize=figsize)
        else:
            self.fig = fig
        if ax is None:
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = ax
        self.node_radius = self.radius*noderadius
        self.ax.set_xlim(-radius*(1+noderadius), radius*(1+noderadius))
        self.ax.set_ylim(-radius*(1+noderadius), radius*(1+noderadius))
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        for k in self.ax.spines.keys():
            self.ax.spines[k].set_visible(False)

    def draw(self):
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        """
        Draws nodes onto the canvas with colours.
        """
        r = self.radius
        node_r = self.node_radius
        # if 'color' in self.nodeprops:
        #    self.nodeprops.pop('color')
        if 'facecolor' in self.nodeprops:
            self.nodeprops.pop('facecolor')
        # Check if self.nodecolor is a string. If so, this color gets applied
        # to all nodes.
        if isinstance(self.nodecolor, str):
            nodes_and_colors = zip(self.nodes,
                                   [self.nodecolor] * len(self.nodes))
        # Check if nodecolor is an iterable. If so and same length as nodes.
        # This applies each matched color to that node.
        elif hasattr(self.nodecolor, '__iter__') and \
                (len(self.nodes) == len(self.nodecolor)):
            nodes_and_colors = zip(self.nodes, self.nodecolor)
        # Throw error if above two conditions are not met.
        else:
            raise TypeError("""nodecolor must be a string or iterable of the
                same length as nodes.""")
        # Draw the nodes to screen.
        for nn, (node, color) in enumerate(nodes_and_colors):
            theta = self.node_theta(node)
            x, y = get_cartesian(r, theta)
            if hasattr(self, 'nodefill'):
                if self.nodefill[nn] == False:
                    self.nodeprops['facecolor'] = [1,1,1]
                else:
                    self.nodeprops['facecolor'] = color
            else:
                self.nodeprops['facecolor'] = color
            self.nodeprops['edgecolor'] = color
            
            node_patch = patches.Ellipse((x, y), node_r, node_r,
                                         lw=0.15, **self.nodeprops)
            self.ax.add_patch(node_patch)

    def node_theta(self, node):
        """
        Maps node to Angle.
        """
        if isinstance(self.nodes, list):
            i = self.nodes.index(node)
        else:
            i = [ll for ll in self.nodes][node]
        theta = i*2*np.pi/len(self.nodes)

        return theta

    def add_edges(self):
        """
        Draws edges to screen.
        """
        for ii, (start, end) in enumerate(self.edges):
            start_theta = self.node_theta(start)
            end_theta = self.node_theta(end)
            verts = [get_cartesian(self.radius, start_theta),
                     (0, 0),
                     get_cartesian(self.radius, end_theta)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]

            path = Path(verts, codes)
            self.edgeprops['facecolor'] = 'none'
            
            if isinstance(self.edgecolor, list) and len(self.edgecolor) > 1:
                self.edgeprops['edgecolor'] = self.edgecolor[ii]
            else:
                self.edgeprops['edgecolor'] = self.edgecolor
            patch = patches.PathPatch(path, lw=self.lw, ls=self.ls, alpha=self.alpha, **self.edgeprops)
            
            self.ax.add_patch(patch)
            

def get_cartesian(r, theta):
    x = r*np.sin(theta)
    y = r*np.cos(theta)

    return x, y
