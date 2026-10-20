from __future__ import annotations
from collections import deque
from pathlib import Path
import random
import networkx as nx

def read_matrix_market(path):
    with Path(path).open(encoding="ascii") as f:
        banner=f.readline().split()
        if len(banner)!=5 or banner[:4]!=["%MatrixMarket","matrix","coordinate","pattern"]: raise ValueError("Expected Matrix Market coordinate pattern")
        symmetry=banner[4].lower()
        if symmetry not in {"symmetric","general"}: raise ValueError("Unsupported symmetry")
        line=f.readline()
        while line.startswith("%"): line=f.readline()
        n,m,declared=map(int,line.split())
        if n!=m: raise ValueError("Matrix must be square")
        g=nx.Graph(); g.add_nodes_from(range(1,n+1)); edges=set(); loops=dupes=count=0
        for lineno,line in enumerate(f,3):
            if not line.strip() or line.startswith("%"): continue
            p=line.split(); u,v=map(int,p[:2]); count+=1
            if not (1<=u<=n and 1<=v<=n): raise ValueError(f"Out of range coordinate at line {lineno}")
            if u==v: loops+=1; continue
            e=min(u,v),max(u,v)
            if e in edges: dupes+=1; continue
            edges.add(e); g.add_edge(*e)
        if count!=declared: raise ValueError(f"Declared {declared} entries, parsed {count}")
    return g,{"format":f"MatrixMarket coordinate pattern {symmetry}","declared_nodes":n,"declared_entries":declared,"parsed_entries":count,"unique_undirected_edges":g.number_of_edges(),"duplicate_edge_entries":dupes,"self_loops":loops,"connected_components":nx.number_connected_components(g),"source":str(path)}

def connected_subgraph(graph,size,seed):
    if size<2 or size>len(graph): raise ValueError(f"size must be between 2 and {len(graph)}")
    rng=random.Random(seed); start=rng.choice(sorted(graph)); order=[]; seen={start}; q=deque([start])
    while q and len(order)<size:
        u=q.popleft(); order.append(u); ns=sorted(graph.neighbors(u)); rng.shuffle(ns)
        for v in ns:
            if v not in seen: seen.add(v); q.append(v)
    if len(order)!=size: raise ValueError(f"Seed component has only {len(order)} nodes")
    ids=sorted(order); mapping={u:i for i,u in enumerate(ids)}; sub=nx.Graph(); sub.add_nodes_from(range(size))
    sub.add_edges_from((mapping[u],mapping[v]) for u,v in graph.subgraph(ids).edges)
    return sub,{"seed":seed,"selection":"seeded breadth-first traversal; induced subgraph","start_original_id":start,"local_to_original":ids,"original_to_local":{str(k):v for k,v in mapping.items()}}

def distance_decreasing_hop(graph,distances,current,destination):
    d=distances.get(current)
    if d is None or current==destination:return None
    choices=[v for v in graph.neighbors(current) if distances.get(v,float("inf"))<d]
    return min(choices,key=lambda v:(distances[v],v)) if choices else None

def route_distance_constrained(graph,source,destination,distances=None):
    if source==destination:return {"status":"delivered","hops":0}
    if distances is None: distances=nx.single_source_shortest_path_length(graph,destination)
    current=source; seen={current}; hops=0
    while current!=destination:
        nxt=distance_decreasing_hop(graph,distances,current,destination)
        if nxt is None:return {"status":"dead_end","hops":hops}
        if nxt in seen:return {"status":"loop","hops":hops+1}
        current=nxt;seen.add(current);hops+=1
    return {"status":"delivered","hops":hops}

