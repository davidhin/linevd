import networkx as nx

# from sastvd.embedding_ids import node_type_map, edge_type_map
import sastvd.linevd as svd
import sastvd.helpers.joern as svdj
import sastvd.helpers.dclass as svddc


def get_edge_subgraph(cpg, graph_etype):
    # graph_etype = edge_type_map[graph_etype]
    filtered_edges = [(u, v, k) for u, v, k, etype in cpg.edges(keys=True, data='type')
                      if etype == graph_etype]
    return cpg.edge_subgraph(edges=filtered_edges)


class ReachingDefinitions:
    def __init__(self, cpg):
        self.cpg = cpg
        # cfg_node_subgraph = cpg.subgraph(nodes=[n for n, is_cfg_node in cpg.nodes(data="isCFGNode") if is_cfg_node])
        # cfg_edge_subgraph = get_subgraph(cpg, edge_type_map['FLOWS_TO'])
        # self.cfg = nx.algorithms.operators.union(cfg_node_subgraph, cfg_edge_subgraph)
        self.cfg = get_edge_subgraph(cpg, 'CFG')
        self.ast = get_edge_subgraph(cpg, 'AST')
        # self.args = get_edge_subgraph(cpg, 'ARGUMENT')

    @property
    def variables(self):
        """
        TODO: Rename to "domain"
        """
        return set(self.cpg.nodes[n]['code'] for n in self.cpg.nodes if
                   self.cpg.nodes[n]['label'] == ["IDENTIFIER"])

    def get_assigned_variable(self, node):
        """Get the name of the variable assigned in the node, if any"""
        if node in self.ast.nodes:
            # assign_descendants = [desc for desc in nx.descendants(self.ast, node)
            #                       if self.cpg.nodes[desc]['name'] == node_type_map['<operator>.assignment']]
            assign_descendants = [desc for desc in nx.descendants(self.ast, node)
                                  if self.cpg.nodes[desc]['name'] == '<operator>.assignment']
            for ass in assign_descendants:
                desc = sorted(nx.descendants(self.args, node), key=lambda n: self.cpg[n]["order"])
                if len(desc) > 0:
                    return desc[0]["code"]
        return None

    def gen(self, node):
        """Generate reaching defs for this node"""
        assigned_variable = self.get_assigned_variable(node)
        if assigned_variable is None:
            return set()
        else:
            return {(assigned_variable, node)}

    def kill(self, node, reaching_defs):
        """Kill reaching defs for this node"""
        assigned_variable = self.get_assigned_variable(node)
        if assigned_variable is None:
            return set()
        else:
            return {rd for rd in reaching_defs if rd[0] == assigned_variable}

    def get_reaching_definitions(self):
        """https://www.cs.cmu.edu/afs/cs/academic/class/15745-s16/www/lectures/L6-Foundations-of-Dataflow.pdf"""
        out_reachingdefs = {}
        for n in self.cfg.nodes():
            out_reachingdefs[n] = set()

        in_reachingdefs = {}
        worklist = list(self.cfg.nodes())
        while len(worklist) > 0:
            n = worklist.pop()
            in_reachingdefs[n] = set()
            for p in self.cfg.predecessors(n):
                in_reachingdefs[n] = in_reachingdefs[n].union(out_reachingdefs[p])

            new_out_reaching_defs = self.gen(n).union((in_reachingdefs[n].difference(self.kill(n, in_reachingdefs[n]))))
            if new_out_reaching_defs != out_reachingdefs[n]:
                for s in self.cfg.successors(n):
                    worklist.append(s)
            out_reachingdefs[n] = new_out_reaching_defs

        return in_reachingdefs
    
    def __str__(self):
        return f'{self.variables} {self.get_reaching_definitions()}'



def get_cpg(id_itempath):
    n, e = svdj.get_node_edges(id_itempath)
    n, e = svd.ne_groupnodes(n, e)
    
    e = svdj.rdg(e, "dataflow")
    n = svdj.drop_lone_nodes(n, e)

    nodes = n
    edges = e

    print('nodes', nodes.columns, nodes.head())
    print('edges', edges.columns, edges.head())
    
    # Run dataflow problem extractor (modify to process new-joern structure)
    cpg = nx.MultiDiGraph()
    cpg.add_nodes_from(nodes.apply(lambda n: (n.id, {'code': n.code, 'name': n.name, 'label': n._label, 'order': n.order}), axis=1))
    cpg.add_edges_from(edges.apply(lambda e: (e.outnode, e.innode, {'type': e.etype}), axis=1))
    # Extract CFG with code
    # cpg.add_nodes_from(dict(zip(nodes.id, [{'code': c, 'name': n, 'label': l, 'order': l} for c, n, l, o in zip(nodes.code, nodes.name, nodes._label, nodes.order)])))
    # cpg.add_edges_from(dict(zip(edges.outnode.tolist(), edges.innode.tolist(), [{'type': c} for c in edges.etype])))

    return cpg

def test_get_cpg():
    cpg = get_cpg(svddc.BigVulDataset.itempath(10))
    print(cpg)
    print(ReachingDefinitions(cpg))