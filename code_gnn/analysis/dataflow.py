import networkx as nx

from code_gnn.models.embedding_ids import node_type_map, edge_type_map


def get_edge_subgraph(cpg, graph_etype):
    graph_etype = edge_type_map[graph_etype]
    filtered_edges = [(u, v, k) for u, v, k, etype in cpg.edges(keys=True, data='type')
                      if etype == graph_etype]
    return cpg.edge_subgraph(edges=filtered_edges)


class ReachingDefinitions:
    def __init__(self, cpg):
        self.cpg = cpg
        # cfg_node_subgraph = cpg.subgraph(nodes=[n for n, is_cfg_node in cpg.nodes(data="isCFGNode") if is_cfg_node])
        # cfg_edge_subgraph = get_subgraph(cpg, edge_type_map['FLOWS_TO'])
        # self.cfg = nx.algorithms.operators.union(cfg_node_subgraph, cfg_edge_subgraph)
        self.cfg = get_edge_subgraph(cpg, 'FLOWS_TO')
        self.ast = get_edge_subgraph(cpg, 'IS_AST_PARENT')

    @property
    def variables(self):
        """
        TODO: Rename to "domain"
        """
        return set(self.cpg.nodes[n]['code'] for n in self.cpg.nodes if
                   self.cpg.nodes[n]['type'] == node_type_map['Identifier'])

    def get_assigned_variable(self, node):
        """Get the name of the variable assigned in the node, if any"""
        if node in self.ast.nodes:
            assign_descendants = [desc for desc in nx.descendants(self.ast, node)
                                  if self.cpg.nodes[desc]['type'] == node_type_map['AssignmentExpression']]
            if len(assign_descendants) > 0:
                assign_expr = assign_descendants[0]
                assign_successors = self.ast.successors(assign_expr)
                assign_successors = [succ for succ in assign_successors
                                     if self.cpg.nodes[succ]['type'] == node_type_map['Identifier']]
                if len(assign_successors) > 0:
                    assigned_variable_id = min(assign_successors, key=lambda s: self.cpg.nodes[s]['childNum'])
                    assigned_variable = self.cpg.nodes[assigned_variable_id]['code']
                    return assigned_variable
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
