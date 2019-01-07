# -*- coding: utf-8 -*-
from time import time
from collections import deque
import numpy as np
import math,logging
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from utils import *
import os

limiteDist = 20

def getDegreeListsVertices(g,vertices,calcUntilLayer):
    degreeList = {}

    for v in vertices:
        degreeList[v] = getDegreeLists(g,v,calcUntilLayer)

    return degreeList

def getCompactDegreeListsVertices(g,vertices,maxDegree,calcUntilLayer,common):
    degreeList = {}
    commonFriendsList = {}

    for v in vertices:
        degreeList[v], commonFriendsList[v] = getCompactDegreeLists(g,v,maxDegree,calcUntilLayer, common)

    return degreeList, commonFriendsList

def getCommonFriendsMed(common,v):
    return common[v]


def getCompactDegreeLists(g, root, maxDegree,calcUntilLayer, common):
    t0 = time()

    listas = {}
    commons = {}
    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    l = {}
    q = deque()
    
    
    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        d = len(g[vertex])
        if(d not in l):
            l[d] = 0
        l[d] += 1

        q.append(getCommonFriendsMed(common,vertex))

        for v in g[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1


        if(timeToDepthIncrease == 0):

            list_d = []
            for degree,freq in l.iteritems():
                list_d.append((degree,freq))
            list_d.sort(key=lambda x: x[0])
            listas[depth] = np.array(list_d,dtype=np.int32)

            qp = np.array(q,dtype='float')
            qp = np.sort(qp)
            commons[depth] = qp


            q = deque()
            l = {}

            if(calcUntilLayer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root,(t1-t0)))

    return listas, commons

def searchCommonFriends(sorted_full,sorted_sub):
    full_ix = 0
    sub_ix = 0
    count = 0
    while full_ix < len(sorted_full) and sub_ix < len(sorted_sub):
        curr_full = sorted_full[full_ix]
        curr_sub = sorted_sub[sub_ix]
        if curr_full == curr_sub:
            count += 1
            full_ix +=1
            sub_ix +=1
            continue
        if curr_full < curr_sub:
            full_ix +=1
        else:
            sub_ix+=1
    return count


def getDegreeLists(g, root, calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    

    l = deque()
    
    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        l.append(len(g[vertex]))

        for v in g[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1    

        if(timeToDepthIncrease == 0):

            lp = np.array(l,dtype='float')
            lp = np.sort(lp)
            listas[depth] = lp
            l = deque()

            if(calcUntilLayer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root,(t1-t0)))


    return listas

def cost(a,b):
    ep = 0.5
    m = max(a,b) + ep
    mi = min(a,b) + ep
    return ((m/mi) - 1)

def cost_min(a,b):
    ep = 0.5
    m = max(a[0],b[0]) + ep
    mi = min(a[0],b[0]) + ep
    return ((m/mi) - 1) * min(a[1],b[1])


def cost_max(a,b):
    ep = 0.5
    m = max(a[0],b[0]) + ep
    mi = min(a[0],b[0]) + ep
    return ((m/mi) - 1) * max(a[1],b[1])

def preprocess_degreeLists():

    logging.info("Recovering degreeList from disk...")
    degreeList = restoreVariableFromDisk('degreeList')

    logging.info("Creating compactDegreeList...")

    dList = {}
    dFrequency = {}
    for v,layers in degreeList.iteritems():
        dFrequency[v] = {}
        for layer,degreeListLayer in layers.iteritems():
            dFrequency[v][layer] = {}
            for degree in degreeListLayer:
                if(degree not in dFrequency[v][layer]):
                    dFrequency[v][layer][degree] = 0
                dFrequency[v][layer][degree] += 1
    for v,layers in dFrequency.iteritems():
        dList[v] = {}
        for layer,frequencyList in layers.iteritems():
            list_d = []
            for degree,freq in frequencyList.iteritems():
                list_d.append((degree,freq))
            list_d.sort(key=lambda x: x[0])
            dList[v][layer] = np.array(list_d,dtype='float')

    logging.info("compactDegreeList created!")

    saveVariableOnDisk(dList,'compactDegreeList')

def verifyDegrees(degrees,degree_v_root,degree_a,degree_b):

    if(degree_b == -1):
        degree_now = degree_a
    elif(degree_a == -1):
        degree_now = degree_b
    elif(abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root)):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now 

def get_vertices(v,degree_v,degrees,a_vertices):
    a_vertices_selected = 2 * math.log(a_vertices,2)
    #logging.info("Selecionando {} próximos ao vértice {} ...".format(int(a_vertices_selected),v))
    vertices = deque()

    try:
        c_v = 0  

        for v2 in degrees[degree_v]['vertices']:
            if(v != v2):
                vertices.append(v2)
                c_v += 1
                if(c_v > a_vertices_selected):
                    raise StopIteration

        if('before' not in degrees[degree_v]):
            degree_b = -1
        else:
            degree_b = degrees[degree_v]['before']
        if('after' not in degrees[degree_v]):
            degree_a = -1
        else:
            degree_a = degrees[degree_v]['after']
        if(degree_b == -1 and degree_a == -1):
            raise StopIteration
        degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

        while True:
            for v2 in degrees[degree_now]['vertices']:
                if(v != v2):
                    vertices.append(v2)
                    c_v += 1
                    if(c_v > a_vertices_selected):
                        raise StopIteration

            if(degree_now == degree_b):
                if('before' not in degrees[degree_b]):
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]['before']
            else:
                if('after' not in degrees[degree_a]):
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]['after']
            
            if(degree_b == -1 and degree_a == -1):
                raise StopIteration

            degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

    except StopIteration:
        #logging.info("Vértice {} - próximos selecionados.".format(v))
        return list(vertices)

    return list(vertices)


def splitDegreeList(part,c,G,compactDegree):
    if(compactDegree):
        logging.info("Recovering compactDegreeList from disk...")
        degreeList = restoreVariableFromDisk('compactDegreeList')
    else:
        logging.info("Recovering degreeList from disk...")
        degreeList = restoreVariableFromDisk('degreeList')

    logging.info("Recovering degree vector from disk...")
    degrees = restoreVariableFromDisk('degrees_vector')

    degreeListsSelected = {}
    vertices = {}
    a_vertices = len(G)

    for v in c:
        nbs = get_vertices(v,len(G[v]),degrees,a_vertices)
        vertices[v] = nbs
        degreeListsSelected[v] = degreeList[v]
        for n in nbs:
            degreeListsSelected[n] = degreeList[n]

    saveVariableOnDisk(vertices,'split-vertices-'+str(part))
    saveVariableOnDisk(degreeListsSelected,'split-degreeList-'+str(part))


   
    logging.info("Recovering commonFriends from disk...")



    


def calc_distances(part, commonList, compactDegree = False):
    vertices = restoreVariableFromDisk('split-vertices-'+str(part))
    degreeList = restoreVariableFromDisk('split-degreeList-'+str(part))

    distances_r = {}
    distances_q = {}

    if compactDegree:
        dist_func = cost_max
    else:
        dist_func = cost
    for v1,nbs in vertices.iteritems():
        lists_v1 = degreeList[v1]
        common_v1 = commonList[v1]

        for v2 in nbs:
            t00 = time()
            lists_v2 = degreeList[v2]
            common_v2 = commonList[v2]
            max_layer = min(len(lists_v1),len(lists_v2))
            distances_r[v1,v2] = {}
            distances_q[v1,v2] = {}
            for layer in range(0,max_layer):
                dist_r, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
                dist_q, path = fastdtw(common_v1[layer],common_v2[layer],radius=1,dist=cost)
                distances_r[v1,v2][layer] = dist_r
                distances_q[v1,v2][layer] = dist_q
            t11 = time()
            logging.info('fastDTW between vertices ({}, {}). Time: {}s'.format(v1,v2,(t11-t00)))

    preprocess_consolides_distances(distances_r)
    preprocess_consolides_distances(distances_q)
    saveVariableOnDisk(distances_r,'distances-r-'+str(part))
    saveVariableOnDisk(distances_q,'distances-q-'+str(part))
    return

def calc_distances_all(vertices,list_vertices,degreeList, commonList, part, compactDegree = False):

    distances_r = {}
    distances_q = {}
    cont = 0

    if compactDegree:
        dist_func = cost_max
    else:
        dist_func = cost

    for v1 in vertices:
        lists_v1 = degreeList[v1]
        common_v1 = commonList[v1]

        for v2 in list_vertices[cont]:
            lists_v2 = degreeList[v2]
            common_v2 = commonList[v2]
            
            max_layer = min(len(lists_v1),len(lists_v2))
            distances_r[v1,v2] = {}
            distances_q[v1,v2] = {}

            for layer in range(0,max_layer):
                #t0 = time()
                dist_r, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
                dist_q, path = fastdtw(common_v1[layer],common_v2[layer],radius=1,dist=cost)
                #t1 = time()
                #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                distances_r[v1,v2][layer] = dist_r
                distances_q[v1,v2][layer] = dist_q
                

        cont += 1

    preprocess_consolides_distances(distances_r)
    preprocess_consolides_distances(distances_q)
    saveVariableOnDisk(distances_r,'distances-r-'+str(part))
    saveVariableOnDisk(distances_q,'distances-q-'+str(part))
    return


def selectVertices(layer,fractionCalcDists):
    previousLayer = layer - 1

    logging.info("Recovering distances from disk...")
    distances = restoreVariableFromDisk('distances')

    threshold = calcThresholdDistance(previousLayer,distances,fractionCalcDists)

    logging.info('Selecting vertices...')

    vertices_selected = deque()

    for vertices,layers in distances.iteritems():
        if(previousLayer not in layers):
            continue
        if(layers[previousLayer] <= threshold):
            vertices_selected.append(vertices)

    distances = {}

    logging.info('Vertices selected.')

    return vertices_selected


def preprocess_consolides_distances(distances, startLayer = 1):

    logging.info('Consolidating distances...')

    for vertices,layers in distances.iteritems():
        keys_layers = sorted(layers.keys())
        startLayer = min(len(keys_layers),startLayer)
        for layer in range(0,startLayer):
            keys_layers.pop(0)


        for layer in keys_layers:
            layers[layer] += layers[layer - 1]

    logging.info('Distances consolidated.')


def exec_bfs_compact(G,workers,calcUntilLayer,common):

    futures = {}
    degreeList = {}
    commonList = {}

    t0 = time()
    vertices = G.keys()
    parts = workers
    chunks = partition(vertices,parts)

    logging.info('Capturing larger degree...')
    maxDegree = 0
    for v in vertices:
        if(len(G[v]) > maxDegree):
            maxDegree = len(G[v])
    logging.info('Larger degree captured')

    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            job = executor.submit(getCompactDegreeListsVertices,G,c,maxDegree,calcUntilLayer, common)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl, ql = job.result()
            v = futures[job]
            degreeList.update(dl)
            commonList.update(ql)

    logging.info("Saving degreeList on disk...")
    saveVariableOnDisk(degreeList,'compactDegreeList')

    logging.info("Saving commonList on disk...")
    saveVariableOnDisk(commonList,'commonList')
    t1 = time()
    logging.info('Execution time - BFS: {}m'.format((t1-t0)/60))


    return


def exec_bfs(G,workers,calcUntilLayer):

    futures = {}
    degreeList = {}

    t0 = time()
    vertices = G.keys()
    parts = workers
    chunks = partition(vertices,parts)

    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            job = executor.submit(getDegreeListsVertices,G,c,calcUntilLayer)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList.update(dl)

    logging.info("Saving degreeList on disk...")
    saveVariableOnDisk(degreeList,'degreeList')
    t1 = time()
    logging.info('Execution time - BFS: {}m'.format((t1-t0)/60))


    return

def exec_common_friends(g):
    res = {}
    for v in g.keys():
        t0 = time()
        med = 0
        neighs = g[v]
        neighs.sort()

        vdegree = float(len(neighs))
        totalcf = 0
        for u in g[v]:
            cf = 0
            neighs_u = g[u]
            neighs_u.sort()
            cf = searchCommonFriends(neighs, neighs_u)
            totalcf += cf
        med = totalcf / vdegree
        t1 = time()
        logging.info('Common friends vertex {}. Med: {}. Time: {}s'.format(v,med,(t1-t0)))
        res[v] = med
    return res

def generate_distances_network_part1(workers):
    parts = workers
    weights_distances_r = {}
    weights_distances_q = {}
    for part in range(1,parts + 1):    
        
        logging.info('Executing part {}...'.format(part))
        distances_r = restoreVariableFromDisk('distances-r-'+str(part))
        distances_q = restoreVariableFromDisk('distances-q-'+str(part))
        
        for vertices,layers in distances_r.iteritems():
            for layer,distance in layers.iteritems():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in weights_distances_r):
                    weights_distances_r[layer] = {}
                weights_distances_r[layer][vx,vy] = distance

        for vertices,layers in distances_q.iteritems():
            for layer,distance in layers.iteritems():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in weights_distances_q):
                    weights_distances_q[layer] = {}
                weights_distances_q[layer][vx,vy] = distance


        logging.info('Part {} executed.'.format(part))

    for layer,values in weights_distances_r.iteritems():
        saveVariableOnDisk(values,'weights_distances-r-layer-'+str(layer))
    for layer,values in weights_distances_q.iteritems():
        saveVariableOnDisk(values,'weights_distances-q-layer-'+str(layer))
    return

def generate_distances_network_part2(workers):
    parts = workers
    graphs_r = {}
    for part in range(1,parts + 1):

        logging.info('Executing part {}...'.format(part))
        distances_r = restoreVariableFromDisk('distances-r-'+str(part))

        for vertices,layers in distances_r.iteritems():
            for layer,distance in layers.iteritems():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in graphs_r):
                    graphs_r[layer] = {}
                if(vx not in graphs_r[layer]):
                   graphs_r[layer][vx] = [] 
                if(vy not in graphs_r[layer]):
                   graphs_r[layer][vy] = [] 
                graphs_r[layer][vx].append(vy)
                graphs_r[layer][vy].append(vx)
        logging.info('Part {} executed.'.format(part))

    for layer,values in graphs_r.iteritems():
        saveVariableOnDisk(values,'graphs-r-layer-'+str(layer))

    return

def generate_distances_network_part3(pcommonf):

    layer = 0
    while(isPickle('graphs-r-layer-'+str(layer))):
        graphs_r = restoreVariableFromDisk('graphs-r-layer-'+str(layer))
        weights_distances_r = restoreVariableFromDisk('weights_distances-r-layer-'+str(layer))
        weights_distances_q = restoreVariableFromDisk('weights_distances-q-layer-'+str(layer))

        
        logging.info('Executing layer {}...'.format(layer))
        alias_method_j = {}
        alias_method_q = {}
        weights = {}
    
        for v,neighbors in graphs_r.iteritems():
            #logging.info('{}'.format(weights_distances_r[v]))
            er_list = deque()
            eq_list = deque()
            sum_wr = 0.0
            sum_wq = 0.0


            for n in neighbors:
                if (v,n) in weights_distances_r:
                    wd = weights_distances_r[v,n]
                else:
                    wd = weights_distances_r[n,v]
                if (v,n) in weights_distances_q:
                    wq = weights_distances_q[v,n]
                else:
                    wq = weights_distances_q[n,v]
                wr = np.exp(-float(wd))
                wrq = np.exp(-float(wq))
                er_list.append(wr)
                eq_list.append(wrq)
                sum_wr += wr
                sum_wq += wrq
            
            er_list = [x / sum_wr for x in er_list]
            eq_list = [x / sum_wq for x in eq_list]
            vec_res = [(va*(1 - pcommonf) + eq_list[i]*pcommonf) for i,va in enumerate(er_list)]
            weights[v] = vec_res
            J, q = alias_setup(vec_res)
            alias_method_j[v] = J
            alias_method_q[v] = q

        saveVariableOnDisk(weights,'distances_nets_weights-layer-'+str(layer))
        saveVariableOnDisk(alias_method_j,'alias_method_j-layer-'+str(layer))
        saveVariableOnDisk(alias_method_q,'alias_method_q-layer-'+str(layer))
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info('Weights created.')

    return


def generate_distances_network_part4():
    logging.info('Consolidating graphs...')
    graphs_c = {}
    layer = 0
    while(isPickle('graphs-r-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        graphs = restoreVariableFromDisk('graphs-r-layer-'+str(layer))
        graphs_c[layer] = graphs
        logging.info('Layer {} executed.'.format(layer))
        layer += 1


    logging.info("Saving distancesNets on disk...")
    saveVariableOnDisk(graphs_c,'distances_nets_graphs')
    logging.info('Graphs consolidated.')
    return

def generate_distances_network_part5():
    alias_method_j_c = {}
    layer = 0
    while(isPickle('alias_method_j-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))          
        alias_method_j = restoreVariableFromDisk('alias_method_j-layer-'+str(layer))
        alias_method_j_c[layer] = alias_method_j
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_j on disk...")
    saveVariableOnDisk(alias_method_j_c,'nets_weights_alias_method_j')

    return

def generate_distances_network_part6():
    alias_method_q_c = {}
    layer = 0
    while(isPickle('alias_method_q-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))          
        alias_method_q = restoreVariableFromDisk('alias_method_q-layer-'+str(layer))
        alias_method_q_c[layer] = alias_method_q
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_q on disk...")
    saveVariableOnDisk(alias_method_q_c,'nets_weights_alias_method_q')

    return

def generate_distances_network(workers, pcommonf):
    t0 = time()
    logging.info('Creating distance network...')

    os.system("rm "+returnPathStruc2vec()+"/../pickles/weights_distances-r-layer-*.pickle")
    os.system("rm "+returnPathStruc2vec()+"/../pickles/weights_distances-q-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part1,workers)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 1: {}s'.format(t))

    t0 = time()
    os.system("rm "+returnPathStruc2vec()+"/../pickles/graphs-r-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part2,workers)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 2: {}s'.format(t))
    logging.info('distance network created.')

    logging.info('Transforming distances into weights...')

    t0 = time()
    os.system("rm "+returnPathStruc2vec()+"/../pickles/distances_nets_weights-layer-*.pickle")
    os.system("rm "+returnPathStruc2vec()+"/../pickles/alias_method_j-layer-*.pickle")
    os.system("rm "+returnPathStruc2vec()+"/../pickles/alias_method_q-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part3, pcommonf)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 3: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part4)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 4: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part5)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 5: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part6)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 6: {}s'.format(t))
 
    return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q
