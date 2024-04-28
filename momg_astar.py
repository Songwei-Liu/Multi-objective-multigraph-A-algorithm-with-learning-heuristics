"""
Paper: A Multi-objective Multigraph A* Algorithm with Online Likely-Admissible Heuristics using Walk-based Shallow Embeddings

@code author: Songwei Liu (songwei.liu@qmul.ac.uk)

Three algorithms are included:
        (a) MOMGA: 
                the version of MOMGA* to cope with admissible heuristics
        (b) MOMGA_LikelyAdmissible: 
                the version of MOMGA* to cope with likely-admissible heuristics
        (c) Backtrack: 
                the backtrack mechanism in MOMGA* to obtain the detailed paths (node IDs with parallel edge IDs)

Input format for MOMGA and MOMGA_LikelyAdmissible:
        G:  
            a multi-objective multigraph with the format requirement of 'networkx' package, where node IDs are strings.

        H:  
            heuristics in dictionary format, with node IDs in string format as keys and heuristics in list format as values.
            e.g., a bi-objective case: H['0'] = [10, 10]

        start_node:
            the origin for the search task, in string format.

        end_node:
            the destination for the search task, in string format.

        objs: 
            the number of optimisation objectives.
"""

import networkx as nx, time


def MOMGA(G, H, start_node, end_node, objs):
    start_time = time.process_time()
    # Some performance monitoring indicators, can be deleted
    iterations = 0 
    expansions = 0
    parallel_edge_expansions = 0
    dom_calls = 0
    
    def F(exact, H): # Calculate the "f" function values, f: approximate total cost
        result = []
        for i, x in enumerate(exact):
            result.append(x + H[i])
        return result
    
    def domination(self, other):
        # Some performance monitoring indicators, can be deleted
        CompareTimes_0 = 0
        CompareTimes_1 = 0
        CompareTimes_2 = 0
        
        not_worse = True
        strictly_better = False
        for x, y in zip(self, other):
            if x > y:
                CompareTimes_0 += 1
                not_worse = False
            elif y > x:
                CompareTimes_1 += 1
                strictly_better = True
            elif x == y:
                CompareTimes_2 += 1
        return not_worse and strictly_better, CompareTimes_0, CompareTimes_1, CompareTimes_2

    ### Finds the set of undominated solutions given a set of solutions updates global front ###
    def global_f(buf2, dom_calls):       # buf2: the set containing alternatives to be checked
        # Some performance monitoring indicators, can be deleted
        Global_dom_calls = 0
        Nondomination = 0
        ComparisonTimes_0 = 0
        ComparisonTimes_1 = 0
        ComparisonTimes_2 = 0

        global_front = []
        for solution_y in buf2:          # solution_y: one element in buf2, (n,g(n),F(n))
            y_is_dominated = False
            for solution_x in buf2:      # solution_x: one element in buf2, (n,g(n),F(n))
                dom_calls += 1
                Global_dom_calls += 1
                AA, BB, CC, DD = domination(solution_x[2], solution_y[2])
                ComparisonTimes_0 += BB
                ComparisonTimes_1 += CC
                ComparisonTimes_2 += DD
                if (AA == True):
                    y_is_dominated = True
                    break
            if (y_is_dominated == False) and (solution_y not in global_front):
                global_front.append(solution_y)
                Nondomination += 1
        return global_front, dom_calls, Global_dom_calls, Nondomination, ComparisonTimes_0, ComparisonTimes_1, ComparisonTimes_2

    zeros = tuple([0 for x in range(objs)])
    SG = nx.DiGraph()
    SG.add_node(start_node)
    G_closed = {}
    G_open = {start_node:[zeros]} #
    Open = [[start_node, zeros, F(zeros, H[str(start_node)])]]

    Costs = []
    
    ### 0 TERMINATION CHECK ###
    while len(Open) > 0: # If Open is empty, then the running is terminated.
        iterations += 1
        ### 1 PATH SELECTION ###
        non_dom, dom_calls, Global_dom_calls, Nondomination, ComparisonTimes_0, ComparisonTimes_1, ComparisonTimes_2 = global_f(Open, dom_calls)
        
        Open_temporary = []
        G_open_temporary = {}
        non_dom_counter = 0

        for non_dom_sub in non_dom:

            non_dom_counter += 1
            Open.remove(non_dom_sub)
            if non_dom_sub[0] in G_closed:
                G_closed[non_dom_sub[0]].append(non_dom_sub[1])
            else:
                G_closed[non_dom_sub[0]] = [non_dom_sub[1]]

            if len(G_open[non_dom_sub[0]]) > 1:
                G_open[non_dom_sub[0]].remove(non_dom_sub[1])
            else:
                del G_open[non_dom_sub[0]]
            
            if non_dom_counter == len(non_dom): # Open_temporary: list; G_Open_temporary: dict.
                for Open_temporary_sub in Open_temporary:
                    if Open_temporary_sub in Open:
                        Open.remove(Open_temporary_sub)

                for aa in G_open_temporary:
                    for bb in G_open_temporary[aa]:
                        if (aa in G_open) and (bb in G_open[aa]):
                            if len(G_open[aa]) > 1:
                                G_open[aa].remove(bb)
                            else:
                                del G_open[aa]
            
            ### SOLUTION RECORDING ###
            if non_dom_sub[0] == end_node:
                Costs.append(non_dom_sub[1])
                Openx = [x for x in Open] # Openx is the copy of Open
                
                for sol in Open:
                    dom_calls += 1
                    if domination(non_dom_sub[1], sol[2])[0] == True: # if True, non_dom_sub[1] is strictly better than sol[2] # Songwei August 2022
                        Openx.remove(sol)
                
                Open = [x for x in Openx]
                continue # should be "continue" rather than "break"
            ### PATH EXPANDING ###
            else:
                for successor in G.neighbors(non_dom_sub[0]):
                    expansions += 1
                    for row_index in range(len(G[non_dom_sub[0]][successor]['c'])):
                        parallel_edge_expansions += 1
                        cost_m = tuple([non_dom_sub[1][i] + G[non_dom_sub[0]][successor]['c'][row_index][i] for i in range(objs)])
                        # If m is a new node
                        if successor not in SG:
                            for cost_sol in Costs:
                                dom_calls += 1
                                if domination(cost_sol, F(cost_m, H[str(successor)]))[0] == True: 
                                    break # If dominated, run "break" and the followed "else" will not be run.
                            else: # If Fm is not empty, put (m, gm, Fm) in Open
                                Open.append([successor, cost_m, F(cost_m,H[str(successor)])])
                                # Set G_open(m) = {gm}
                                if successor in G_open:
                                    G_open[successor].append(cost_m)
                                else:
                                    G_open[successor] = [cost_m]
                                # Label with gm a pointer to n
                                SG.add_edge(successor, non_dom_sub[0], c=[[cost_m, row_index]]) # row_index is the edge ID
                                
                        else:
                        # Elif#1 gm equals some cost vector in G_open(m) U G_cl(m), then label with gm a pointer to n
                        # Elif#2 gm is not dominated by any cost vector in G_open(m) U G_cl(m)
                            grouped = []
                            if successor in G_open:
                                grouped += G_open[successor]
                                if cost_m in G_open[successor]:
                                    if (successor, non_dom_sub[0]) in [(e[0],e[1]) for e in SG.edges()]:
                                        SG[successor][non_dom_sub[0]]['c'].append([cost_m, row_index])
                                    else:
                                        SG.add_edge(successor, non_dom_sub[0], c=[[cost_m, row_index]])
                            if successor in G_closed:
                                grouped += G_closed[successor]
                                if cost_m in G_closed[successor]:
                                    if (successor, non_dom_sub[0]) in [(e[0],e[1]) for e in SG.edges()]:
                                        SG[successor][non_dom_sub[0]]['c'].append([cost_m, row_index])
                                    else:
                                        SG.add_edge(successor, non_dom_sub[0], c=[[cost_m, row_index]])
                            for vector in grouped: 
                                dom_calls += 1
                                if domination(vector, cost_m)[0] == True: # Return false when set1 is better # Songwei August 2022
                                    ErrorJudge = 1
                                    break
                            else:
                                if 'ErrorJudge' not in locals():
                                    if successor in G_open:
                                        G_openXX = [xx for xx in G_open[successor]]
                                        for vector1 in G_open[successor]:
                                            dom_calls += 1
                                            if domination(cost_m, vector1)[0] == True: # Being ture means cost_m is stricly better than vector1 --Songwei August 2022
                                                if non_dom_counter < len(non_dom):
                                                    if successor in G_open_temporary:
                                                        G_open_temporary[successor].append(vector1)
                                                    else:
                                                        G_open_temporary[successor] = [vector1]
                                                    Open_temporary.append([successor, vector1, F(vector1, H[str(successor)])])
                                                elif non_dom_counter == len(non_dom):
                                                    G_openXX.remove(vector1)
                                                    if [successor, vector1, F(vector1, H[str(successor)])] in Open:
                                                        Open.remove([successor, vector1, F(vector1, H[str(successor)])])
                                        if G_openXX == []:
                                            del G_open[successor]
                                        else:
                                            G_open[successor] = [xx for xx in G_openXX]
                                    if successor in G_closed:
                                        G_closedXX = [yy for yy in G_closed[successor]]
                                        for vector2 in G_closed[successor]:
                                            dom_calls += 1
                                            if domination(cost_m, vector2)[0] == True:
                                                G_closedXX.remove(vector2)
                                        if G_closedXX == []:
                                            del G_closed[successor]
                                        else:
                                            G_closed[successor] = [yy for yy in G_closedXX]
                                    for cost_sol in Costs:
                                        dom_calls += 1
                                        if domination(cost_sol, F(cost_m, H[str(successor)]))[0] == True:
                                            break
                                    else:
                                        Open.append([successor, cost_m, F(cost_m, H[str(successor)])])
                                        if successor in G_open: 
                                            G_open[successor].append(cost_m)
                                        else:
                                            G_open[successor] = [cost_m]
                                        if (successor, non_dom_sub[0]) in [(e[0],e[1]) for e in SG.edges()]:
                                            SG[successor][non_dom_sub[0]]['c'].append([cost_m, row_index])
                                        else:
                                            SG.add_edge(successor, non_dom_sub[0], c=[[cost_m, row_index]])
                            if 'ErrorJudge' in locals():
                                del ErrorJudge
                else:
                    continue
    return SG, Costs, expansions, parallel_edge_expansions, dom_calls, iterations


def MOMGA_LikelyAdmissible(G, H, start_node, end_node, objs):
    start_time = time.process_time()
    # Some performance monitoring indicators, can be deleted
    iterations = 0 
    expansions = 0
    parallel_edge_expansions = 0
    dom_calls = 0
        
    def F(exact, H): # Calculate the "f" function values, f: approximate total cost
        result = []
        for i, x in enumerate(exact):
            result.append(x + H[i])
        return result
    
    def domination(self, other): # If ture, "self" is strictly better than "other". Otherwise returns false. --Songwei
        # Some performance monitoring indicators, can be deleted
        CompareTimes_0 = 0
        CompareTimes_1 = 0
        CompareTimes_2 = 0
        
        not_worse = True
        strictly_better = False
        for x, y in zip(self, other): 
            if x > y:
                CompareTimes_0 += 1
                not_worse = False
            elif y > x:
                CompareTimes_1 += 1
                strictly_better = True
            elif x == y:
                CompareTimes_2 += 1
        return not_worse and strictly_better, CompareTimes_0, CompareTimes_1, CompareTimes_2
    
    ### Finds the set of undominated solutions given a set of solutions updates global front ###
    def global_f(buf2, dom_calls):       # buf2: the set containing alternatives to be checked
        # Some performance monitoring indicators, can be deleted
        Global_dom_calls = 0
        Nondomination = 0
        ComparisonTimes_0 = 0
        ComparisonTimes_1 = 0
        ComparisonTimes_2 = 0
        
        global_front = []
        for solution_y in buf2:          # solution_y: one element in buf2, (n,g(n),F(n))
            y_is_dominated = False
            for solution_x in buf2:      # solution_x: one element in buf2, (n,g(n),F(n))
                dom_calls += 1
                Global_dom_calls += 1
                AA, BB, CC, DD = domination(solution_x[2], solution_y[2])
                ComparisonTimes_0 += BB
                ComparisonTimes_1 += CC
                ComparisonTimes_2 += DD
                if (AA == True):
                    y_is_dominated = True
                    break
            if (y_is_dominated == False) and (solution_y not in global_front):
                global_front.append(solution_y)
                Nondomination += 1
        return global_front, dom_calls, Global_dom_calls, Nondomination, ComparisonTimes_0, ComparisonTimes_1, ComparisonTimes_2

    zeros = tuple([0 for x in range(objs)])
    SG = nx.DiGraph()
    SG.add_node(start_node)
    G_closed = {}
    G_open = {start_node:[zeros]} # G_open = {start_node: [(0, 0)]}, here 2 objectives, G_open is dict -- Songwei Nov 2021
    Open = [[start_node, zeros, F(zeros, H[str(start_node)])]] # start_node is str

    Costs = []
    
    ### 0 TERMINATION CHECK ###
    while len(Open) > 0: # If Open is empty, then the running is terminated.
        iterations += 1
        ### 1 PATH SELECTION ###
        
        non_dom, dom_calls, Global_dom_calls, Nondomination, ComparisonTimes_0, ComparisonTimes_1, ComparisonTimes_2 = global_f(Open, dom_calls)

        Open_temporary = []
        G_open_temporary = {}
        non_dom_counter = 0


        for non_dom_sub in non_dom:
            non_dom_counter += 1
            Open.remove(non_dom_sub)
            if non_dom_sub[0] in G_closed:
                G_closed[non_dom_sub[0]].append(non_dom_sub[1])
            else:
                G_closed[non_dom_sub[0]] = [non_dom_sub[1]]

            if len(G_open[non_dom_sub[0]]) > 1:
                G_open[non_dom_sub[0]].remove(non_dom_sub[1])
            else:
                del G_open[non_dom_sub[0]]
            
            if non_dom_counter == len(non_dom): # Open_temporary: list; G_Open_temporary: dict.
                for Open_temporary_sub in Open_temporary:
                    if Open_temporary_sub in Open:
                        Open.remove(Open_temporary_sub)

                for aa in G_open_temporary:
                    for bb in G_open_temporary[aa]:
                        if (aa in G_open) and (bb in G_open[aa]):
                            if len(G_open[aa]) > 1:
                                G_open[aa].remove(bb)
                            else:
                                del G_open[aa]
            
            ### SOLUTION RECORDING ###
            if non_dom_sub[0] == end_node:
                Costs.append(non_dom_sub[1])
                Openx = [x for x in Open] # Openx is the copy of Open
                
                for sol in Open:
                    dom_calls += 1
                    if domination(non_dom_sub[1], sol[2])[0] == True: # if True, non_dom_sub[1] is strictly better than sol[2]
                        Openx.remove(sol)
                Open = [x for x in Openx]
                continue # should be "continue" rather than "break"
            ### PATH EXPANDING ###
            else:
                for successor in G.neighbors(non_dom_sub[0]):
                    expansions += 1
                    for row_index in range(len(G[non_dom_sub[0]][successor]['c'])):
                        parallel_edge_expansions += 1
                        cost_m = tuple([non_dom_sub[1][i] + G[non_dom_sub[0]][successor]['c'][row_index][i] for i in range(objs)])
                        # If m is a new node
                        if successor not in SG:
                            for cost_sol in Costs:
                                dom_calls += 1
                                if domination(cost_sol, F(cost_m, H[str(successor)]))[0] == True:
                                    break # If dominated, run "break" and the followed "else" will not be run.
                            else: # If Fm is not empty, put (m, gm, Fm) in Open
                                Open.append([successor, cost_m, F(cost_m,H[str(successor)])])
                                # Set G_open(m) = {gm}
                                if successor in G_open:
                                    G_open[successor].append(cost_m)
                                else:
                                    G_open[successor] = [cost_m]
                                # Label with gm a pointer to n
                                SG.add_edge(successor, non_dom_sub[0], c=[[cost_m, row_index]]) # row_index is the edge ID
                        else:
                        # Elif#1 gm equals some cost vector in G_open(m) U G_cl(m), then label with gm a pointer to n
                        # Elif#2 gm is not dominated by any cost vector in G_open(m) U G_cl(m)
                            grouped = []
                            if successor in G_open:
                                grouped += G_open[successor]
                                if cost_m in G_open[successor]:
                                    if (successor, non_dom_sub[0]) in [(e[0],e[1]) for e in SG.edges()]:
                                        SG[successor][non_dom_sub[0]]['c'].append([cost_m, row_index])
                                    else:
                                        SG.add_edge(successor, non_dom_sub[0], c=[[cost_m, row_index]])
                            if successor in G_closed:
                                grouped += G_closed[successor]
                                if cost_m in G_closed[successor]:
                                    if (successor, non_dom_sub[0]) in [(e[0],e[1]) for e in SG.edges()]:
                                        SG[successor][non_dom_sub[0]]['c'].append([cost_m, row_index])
                                    else:
                                        SG.add_edge(successor, non_dom_sub[0], c=[[cost_m, row_index]])
                            for vector in grouped: 
                                dom_calls += 1
                                if domination(vector, cost_m)[0] == True: # Return false when set1 is better
                                    ErrorJudge = 1
                                    break
                            else:
                                if 'ErrorJudge' not in locals():
                                    if successor in G_open:
                                        G_openXX = [xx for xx in G_open[successor]] # 7.June.2022 Songwei
                                        for vector1 in G_open[successor]:
                                            dom_calls += 1
                                            if domination(cost_m, vector1)[0] == True: # Being ture means cost_m is stricly better than vector1 --Songwei 12 August 2022
                                                
                                                if non_dom_counter < len(non_dom):
                                                    if successor in G_open_temporary:
                                                        G_open_temporary[successor].append(vector1)
                                                    else:
                                                        G_open_temporary[successor] = [vector1]
                                                    Open_temporary.append([successor, vector1, F(vector1, H[str(successor)])])
                                                elif non_dom_counter == len(non_dom):
                                                    G_openXX.remove(vector1)
                                                    if [successor, vector1, F(vector1, H[str(successor)])] in Open:
                                                        Open.remove([successor, vector1, F(vector1, H[str(successor)])])
                                        if G_openXX == []:
                                            del G_open[successor]
                                        else:
                                            G_open[successor] = [xx for xx in G_openXX]
                                    if successor in G_closed:
                                        G_closedXX = [yy for yy in G_closed[successor]]
                                        for vector2 in G_closed[successor]:
                                            dom_calls += 1
                                            if domination(cost_m, vector2)[0] == True:
                                                G_closedXX.remove(vector2)
                                        if G_closedXX == []:
                                            del G_closed[successor]
                                        else:
                                            G_closed[successor] = [yy for yy in G_closedXX]
                                    for cost_sol in Costs:
                                        dom_calls += 1
                                        if domination(cost_sol, F(cost_m, H[str(successor)]))[0] == True: 
                                            break
                                    else:
                                        Open.append([successor, cost_m, F(cost_m, H[str(successor)])])
                                        if successor in G_open: 
                                            G_open[successor].append(cost_m)
                                        else:
                                            G_open[successor] = [cost_m]
                                        if (successor, non_dom_sub[0]) in [(e[0],e[1]) for e in SG.edges()]:
                                            SG[successor][non_dom_sub[0]]['c'].append([cost_m, row_index])
                                        else:
                                            SG.add_edge(successor, non_dom_sub[0], c=[[cost_m, row_index]])
                            if 'ErrorJudge' in locals():
                                del ErrorJudge
                else:
                    continue
    # Compulsory domination check -- Added on 18.May.2022
    Costs2Delete = []
    for cost_1 in Costs:
        for cost_2 in Costs:
            AAA, BBB, CCC, DDD = domination(cost_2, cost_1)
            if AAA == True: # If ture, cost_2 is strictly better than cost_1
                Costs2Delete.append(cost_1)
                break
    CostsX = [xyz for xyz in Costs]
    for cost_value in Costs2Delete:
        CostsX.remove(cost_value)
    Costs = [xyz for xyz in CostsX]
    return SG, Costs, expansions, parallel_edge_expansions, dom_calls, iterations


def Backtrack(SG, G, start_node, end_node, Costs): 
    def sumSubstract(x,y): 
        return sum([abs(x[i]-y[i]) for i in range(len(x))])
    
    Paths = []
    Edges = []
    for cost in Costs: 
        SubPath = []
        SubEdge = []
        CurrentNode = end_node
        CurrentCost = cost
        SubPath.append(CurrentNode)
        while CurrentNode != start_node:
            Judger = False
            for successorx in SG.neighbors(CurrentNode):
                for labels in SG[CurrentNode][successorx]['c']:
                    if abs(sumSubstract(CurrentCost, labels[0])) < 0.1:
                        Judger = True
                        CurrentCost = tuple([CurrentCost[i] - G[CurrentNode][successorx]['c'][labels[1]][i] for i in range(len(CurrentCost))])
                        CurrentNode = successorx
                        SubEdge.append(labels[1])
                        SubPath.append(CurrentNode)
                        break
                if Judger == True:
                    break
        Paths.append(SubPath)
        Edges.append(SubEdge)
    return Paths, Edges