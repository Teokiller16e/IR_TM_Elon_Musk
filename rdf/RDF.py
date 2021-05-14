from rdflib import Graph, Literal, RDF, URIRef, namespace
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import matplotlib.pyplot as plt 
import networkx as nx
import io
import pydotplus
from IPython.display import display, Image
from rdflib.tools.rdf2dot import rdf2dot

def visualize(g):
    stream = io.StringIO()
    rdf2dot(g, stream, opts = {display})
    dg = pydotplus.graph_from_dot_data(stream.getvalue())
    png = dg.create_png()
    display(Image(png))

# More information query commands SELECT,CONSTRUCT,DESCRIBE,ASK
# https://medium.com/@atakanguney94/introduction-to-resource-description-framework-and-sparql-rdf-101-5857f4a6a8a6 

# TODO : networkx plugin--> a) Remove duplicate Elon Musk, b) Shrink entities, c)Check turtle format to remove greek / probably not necessary

graph = Graph()

elon = URIRef('http://dbpedia.org/resource/Elon_Musk')
space_x = URIRef('http://dbpedia.org/resource/SpaceX')
tesla = URIRef('http://dbpedia.org/resource/Tesla')
nasa = URIRef('http://dbpedia.org/resource/NASA')
openAI = URIRef('https://dbpedia.org/page/OpenAI')

graph.add((elon,RDF.type,namespace.FOAF.Person))
graph.add((elon,namespace.DCTERMS['founder'],tesla))
graph.add((elon,namespace.DCTERMS['founder'],space_x))


graph.serialize(format='turtle').decode('u8')

G = rdflib_to_networkx_multidigraph(graph)

# Plot Networkx instance of RDF Graph
pos = nx.spring_layout(G, scale=2,seed=14)
edge_labels = nx.get_edge_attributes(G, 'r')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
nx.draw(G, with_labels=True)

#if not in interactive mode for 
plt.show()


