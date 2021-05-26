from base64 import encode
from typing_extensions import final
from networkx.algorithms.distance_measures import center
from rdflib import Graph, Literal, RDF, URIRef, namespace
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import matplotlib.pyplot as plt 
import networkx as nx
import io
import pydotplus
from IPython.display import display, Image
from rdflib.tools.rdf2dot import rdf2dot
import pandas as pd 
import numpy as np

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


graph.parse(elon)
graph.serialize(format='turtle').decode('u8')

final_graph = Graph()
#final_graph.add((elon,RDF.type,namespace.FOAF.Person))

subjects = []
preds = []
objs = []

# hand coded preprocessing
removed_words = ['wikiPageWikiLink','wikiPageUsesTemplate','22-rdf-syntax-ns#type','wikiPageRedirects',
'owl#sameAs','rdf-schema#comment','rdf-schema#label','abstract','wikiPageDisambiguates','wikiPageLength','prov#wasDerivedFrom','personFunction','subject',
'wikiPageRevisionID','isPrimaryTopicOf','wikiPageExternalLink','wikt','primaryTopic','b','v','voy','float','c','s','n']

elon_family = ['spouse','partner','children','mother','child','relative','father','predecessor']
elon_founded = ['owner','founder','foundedBy','occupation','title','activeYearsStartYear','education','owningCompany','producer','designer','shipOwner']

elon_personal = ['citizenship','birthPlace','birthDate','networth','birthYear','signature','thumbnail','yearsActive','depiction','signatureAlt','wikiPageID']

for index, (sub,predicate,object) in enumerate(graph):
    if object.split('/')[-1] == '' or sub.split('/')[-1]== '' :
        continue
    if (object.split('/')[-1] == 'Elon_Musk' or sub.split('/')[-1]== 'Elon_Musk') and predicate.split('/')[-1]  not in removed_words: 
        subjects.append(URIRef(sub.split('/')[-1]))
        preds.append(URIRef(predicate.split('/')[-1]))
        obja = object.split('/')[-1]
        objs.append(URIRef(obja.split(':')[-1]))

        #sub.append(URIRef(sub.split('/')[-1]))
        #predicate.append(URIRef(predicate.split('/')[-1]))
        #object.append(URIRef(object.split('/')[-1]))

        #final_graph.add((sub,predicate,object))


#source = [i[0] for i in subjects]

# extract object
#target = [i[1] for i in objs]

kg_df = pd.DataFrame({'source':subjects, 'target':objs, 'edge':preds})
kg_df.drop_duplicates()
print(kg_df)


#graph.add((elon,RDF.type,namespace.FOAF.Person))
#graph.add((elon,namespace.DCTERMS['founder'],tesla))
#graph.add((elon,namespace.DCTERMS['founder'],space_x))


print(kg_df.loc[kg_df['target']=="Elon_Musk"])
# create a directed-graph from a dataframe
G=nx.from_pandas_edgelist(kg_df,"source", "target", edge_attr="edge", create_using=nx.MultiDiGraph())

edge_labels = nx.get_edge_attributes(G,'label')
pos = nx.spring_layout(G,scale=10) # k regulates the distance between nodes
#pos = nx.circular_layout(G,scale=100)
#pos = nx.planar_layout(G,scale=5)
#pos = nx.shell_layout(G,scale=100)
#labels = {e: G.edges[e]['edge'] for e in G.edges}

labels = { (subjects[idx],objs[idx]):pred for idx,pred in enumerate(preds) }

nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
nx.draw(G, with_labels=True, node_color='yellow', node_size=750, edge_cmap=plt.cm.Blues,pos=pos)

plt.title("Preprocessed and removed unecessary predicates")
plt.show()


