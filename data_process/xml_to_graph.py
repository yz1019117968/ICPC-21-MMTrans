#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: Zhen YANG
# created at:27/11/2020 1:29 PM
# contact: zhyang8-c@my.cityu.edu.hk

from html.parser import HTMLParser
import networkx as nx
import numpy as np
from data_process.utils import re_0001_, re_0002, re_opt


class MyHTMLParser(HTMLParser):
    def __init__(self):
        super(MyHTMLParser, self).__init__()
        self.parentstack = list()
        self.curtag = -1
        self.tagidx = -1
        self.graph = nx.Graph()
        self.seq = list()

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        self.parentstack.append(self.curtag)
        self.tagidx += 1
        self.seq.append(tag)
        self.graph.add_node(self.tagidx, text=tag)
        if self.parentstack[-1] >= 0:
            self.graph.add_edge(self.parentstack[-1], self.tagidx)
        self.curtag = self.tagidx

    def handle_endtag(self, tag):
        self.curtag = self.parentstack.pop()


    def handle_data(self, data):
        # first, do data text preprocessing
        if re_opt.fullmatch(data) is None and data != "$NUM$" and data != "$STR$" and data != "$ADDR$":
            data = re_0001_.sub(re_0002, data).strip()
        if data == "$NUM$" or data == "$STR$" or data == "$ADDR$":
            data = "「" + data[1:-1] + "」"
        else:
            data = data.lower()
        # second, create a node if there is text
        if(data != ''):
            for d in data.split(' '): # each word gets its own node
                if d != '':
                    self.parentstack.append(self.curtag)
                    self.tagidx += 1
                    self.seq.append(d)
                    self.graph.add_node(self.tagidx, text=d)
                    self.graph.add_edge(self.parentstack[-1], self.tagidx)
                    self.curtag = self.tagidx
                    self.curtag = self.parentstack.pop()

    def get_graph(self):
        return(self.graph)

    def get_seq(self):
        return(self.seq)

def xmldecode(unit):
    parser = MyHTMLParser()
    parser.feed(unit)
    return(parser.get_graph(), parser.get_seq())

def xml_graph(contract_folder, xml_file):
    with open("../contracts/contracts_seqs_xml/{}/{}".format(contract_folder, xml_file), "r", encoding="utf-8") as fr:
    # with open("../contracts/test_xml.txt", "r", encoding="utf-8") as fr:
        text = ""
        line = fr.readline()
        while line:
            text += line
            line = fr.readline()
        (graph, _) = xmldecode(text)
        try:
            nodes = list(graph.nodes.data())
            edges = nx.adjacency_matrix(graph)
        except:
            eg = nx.Graph()
            eg.add_node(0)
            nodes = np.asarray([0])
            edges = nx.adjacency_matrix(eg)
        nodes = " ".join([node[1]['text'] for node in nodes])
        return nodes, edges

if __name__ == "__main__":
    nodes, edges = xml_graph("contract1", "44_35.txt")
    print(nodes)
    print(edges.todense())


