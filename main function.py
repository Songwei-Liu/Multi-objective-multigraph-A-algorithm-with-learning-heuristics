import networkx as nx, math, xlrd, os, time
from momg_astar import MOMGA, Backtrack

def load_multigraph(SglOglG, XlsPath, XlsName, number_of_objectives):
    start_time = time.process_time()
    # Load the cost matrix from excel file
    FilePath = os.path.join(XlsPath, XlsName)
    File = xlrd.open_workbook(FilePath)
    Sheet1_object = File.sheet_by_index(0)

    # Load the node connection relation
    newG = nx.DiGraph(SglOglG)
    newG = nx.DiGraph() # Now newG is a empty graph but the edge information has been in edgelist. -Songwei

    # Convert to a multigraph
    for Row in range(0, Sheet1_object.nrows):
        Col_Num = 0
        g = []
        for Col in range(0, Sheet1_object.ncols):       # Even the number of columns is different among rows, ".nclos" returns the maximum number.
            if Sheet1_object.cell_type(Row, Col) != 0:  # cell_type == 0: empty
                Col_Num += 1
        for ParallelEdge_Counter in range(0, int((Col_Num-2)/number_of_objectives)):
            costvector = []
            for Obj_Counter in range(0, number_of_objectives):
                costvector.append(int(Sheet1_object.cell_value(Row, 2+ParallelEdge_Counter*number_of_objectives+Obj_Counter))) # Has been tested
            g.append(tuple(costvector))
        newG.add_edge(str(Sheet1_object.cell_value(Row,0)), str(Sheet1_object.cell_value(Row,1)), c=g)
    run_time = time.process_time() - start_time
    return newG, run_time

def run_momga(G,O,D,obj,time_limit):
    """
    Given a multigraph G with cost information, origin node O, destination node D, 
    number of objectives obj, executes the NAMOA* algorithm for multigraphs.
    A heuristic function H can be specified here.
    """
    H = {}
    for node in G:
        H[node] = [0 for x in range(obj)]    # There is NO heuristic function! --Songwei
    start_time = time.process_time()
    SG,Costs,exp,parallel_edge_expansions,dom,ite = MOMGA(G, H, O, D, obj)
    rtime = time.process_time() - start_time
    sol = len(Costs)
    
    start_time_BT = time.process_time()
    paths,index_lists = Backtrack(SG, G, O, D, Costs)
    rtime_BT = time.process_time() - start_time_BT

    return rtime, sol, dom, exp, parallel_edge_expansions, ite, Costs, paths, index_lists, rtime_BT

if __name__ == "__main__":
    number_of_objectives = 2
    max_parallel = 10
    # No need to change the above "number_of_objectives" and "max_parallel"

    a = 1500                 # numbe of nodes
    obj_correlation = 'Nega' # or 'Posi'
    O = '3'
    D = '750'
    
    ReadPath = r'./benchmark multi-objective multigraphs/' + str(a) + '_' + obj_correlation + '/'
    SmpfG_Name = 'SmpfG_Obj0.weighted.edgelist'  # Obj0 and Obj1 both work
    XlsName_Read = 'G_Edges_Nodes'+str(a)+'Obj'+str(number_of_objectives)+'P'+str(max_parallel)+'.xls'
    SglFile = os.path.join(ReadPath, SmpfG_Name)
    SglOglG = nx.read_weighted_edgelist(SglFile) # Single-objective (Objective_id) original graph
    ### Change parameters here! | The saved result ###
    OutputPath = r'./'
    TxtName = 'Solutions_NumNodes'+str(a)+'_Test.txt'
    TxtFilePath = os.path.join(OutputPath,TxtName)
    ##################################################
    
    G_multi, G_time = load_multigraph(SglOglG, ReadPath, XlsName_Read, number_of_objectives)

    Momga_run_time,Momga_solutions,Momga_domination_checks,Momga_exppantions,Momga_parallel_edge_expansions,Momga_iterations,Momga_costs,Momga_paths_strList,Momga_index_lists,Momga_rtime_BT = run_momga(G_multi,O,D,number_of_objectives,time_limit=None)
    
    with open(TxtFilePath,"a") as f:
        
        f.write('Running time of MOMGA* was - seconds:\n')
        f.write(str(Momga_run_time)+'\n\n')
        
        f.write('Solutions of MOMGA*:\n')
        f.write(str(Momga_costs)+'\n\n')
        
        f.write('Paths as lists of node IDs in MOMGA*:\n')
        f.write(str(Momga_paths_strList)+'\n\n')
        
        f.write('Paths as lists of parallel edges in MOMGA*:\n')
        f.write(str(Momga_index_lists)+'\n\n')
        