from rdflib import Graph, Literal, RDF, URIRef, namespace
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import matplotlib.pyplot as plt 
import networkx as nx

# More information query commands SELECT,CONSTRUCT,DESCRIBE,ASK
# https://medium.com/@atakanguney94/introduction-to-resource-description-framework-and-sparql-rdf-101-5857f4a6a8a6 

# TODO : networkx plugin--> a) Remove duplicate Elon Musk, b) Shrink entities, c)Check turtle format to remove greek / probably not necessary

graph = Graph()
result = graph.parse('http://dbpedia.org/resource/Elon_Musk',format='turtle')

final_graph = rdflib_to_networkx_multidigraph(result)

# Number of nodes 1296 and number of edges 1564:
print(len(final_graph)) # nodes
print(final_graph.number_of_edges()) # edges

# 1920 * 1080 / 1296 = 1600 --> sqrt(1600) = 40 pixels for each of the graph nodes

# Plot Networkx instance of RDF Graph:
pos = nx.spring_layout(final_graph,scale=2,seed=1)
edge_labels = nx.get_edge_attributes(final_graph,'r')
nx.draw_networkx_edge_labels(final_graph,pos,edge_labels)
#nx.draw_network_edge_labels(final_graph,pos,edge_labels)

nx.draw(final_graph,with_labels=True,node_size=1)

plt.show()

# Iterating through all subjects, predicates and objects
for index,(sub, pred, obj) in enumerate(graph):
    print(sub)
    if index == 20:
        break

#print(f' The graph has {len(graph)} facts')

# Print out the entire RDF graph in turtle format:
#print(graph.serialize(format= 'ttl').decode('u8'))

"""
# new :
graph2 = Graph()
mason = URIRef("http://example.org/mason")

# Add triples using store's add() method.
graph2.add((mason,RDF.type,namespace.FOAF.Person))
graph2.add((mason,namespace.FOAF.nick, Literal("mason",lang="en")))
graph2.add((mason,namespace.FOAF.name,Literal("Mason Carter")))
graph2.add((mason,namespace.FOAF.mbox,URIRef("mailto:mason@example.org")))

"""
