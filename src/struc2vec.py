# -*- coding: utf-8 -*-

import numpy as np
import random,sys,logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from time import time
from collections import deque

from utils import *
from algorithms import *
from algorithms_distances import *
import graph


class Graph():
	def __init__(self, g, is_directed, workers, untilLayer = None, pcommonf = 0.5):

		logging.info(" - Converting graph to dict...")
		self.G = g.gToDict()
		logging.info("Graph converted.")

		self.num_vertices = g.number_of_nodes()
		self.num_edges = g.number_of_edges()
		self.is_directed = is_directed
		self.workers = workers
		self.pcommonf = pcommonf
		self.calcUntilLayer = untilLayer
		logging.info('Graph - Number of vertices: {}'.format(self.num_vertices))
		logging.info('Graph - Number of edges: {}'.format(self.num_edges))
		self.common = {}

	def prepareCommonFriends(self):
    #t0 = time()
		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(exec_common_friends,self.G)
			a = job.result()
			self.common.update(a)
		return

	def preprocess_neighbors_with_bfs(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(exec_bfs,self.G,self.workers,self.calcUntilLayer)
			
			job.result()

		return

	def preprocess_neighbors_with_bfs_compact(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(exec_bfs_compact,self.G,self.workers,self.calcUntilLayer, self.common)
			
			job.result()

		return

	def preprocess_degree_lists(self):

		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(preprocess_degreeLists)
			
			job.result()

		return


	def create_vectors(self):
		logging.info("Creating degree vectors...")
		degrees = {}
		degrees_sorted = set()
		G = self.G
		for v in G.keys():
			degree = len(G[v])
			degrees_sorted.add(degree)
			if(degree not in degrees):
				degrees[degree] = {}
				degrees[degree]['vertices'] = deque() 
			degrees[degree]['vertices'].append(v)
		degrees_sorted = np.array(list(degrees_sorted),dtype='int')
		degrees_sorted = np.sort(degrees_sorted)

		l = len(degrees_sorted)
		for index, degree in enumerate(degrees_sorted):
			if(index > 0):
				degrees[degree]['before'] = degrees_sorted[index - 1]
			if(index < (l - 1)):
				degrees[degree]['after'] = degrees_sorted[index + 1]
		logging.info("Degree vectors created.")
		logging.info("Saving degree vectors...")
		saveVariableOnDisk(degrees,'degrees_vector')


	def calc_distances_all_vertices(self,compactDegree = False):

		logging.info("Using compactDegree: {}".format(compactDegree))
		if(self.calcUntilLayer):
			logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

		futures = {}

		count_calc = 0

		vertices = list(reversed(sorted(self.G.keys())))

		if(compactDegree):
		    logging.info("Recovering degreeList from disk...")
		    degreeList = restoreVariableFromDisk('compactDegreeList')
		else:
		    logging.info("Recovering compactDegreeList from disk...")
		    degreeList = restoreVariableFromDisk('degreeList')

		logging.info("Recovering commonList from disk...")
		commonList = restoreVariableFromDisk('commonList')	

		parts = self.workers
		chunks = partition(vertices,parts)

		t0 = time()

		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part))
				list_v = []
				for v in c:
					list_v.append([vd for vd in degreeList.keys() if vd > v])
				job = executor.submit(calc_distances_all, c, list_v, degreeList, commonList,part, compactDegree = compactDegree)
				futures[job] = part
				part += 1


			logging.info("Receiving results...")

			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} Completed.".format(r))
		
		logging.info('Distances calculated.')
		t1 = time()
		logging.info('Time : {}m'.format((t1-t0)/60))
		
		return


	def calc_distances(self, compactDegree = False):

		logging.info("Using compactDegree: {}".format(compactDegree))
		if(self.calcUntilLayer):
			logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

		futures = {}
		#distances = {}

		count_calc = 0

		G = self.G
		vertices = G.keys()

		parts = self.workers
		chunks = partition(vertices,parts)
		commonList = restoreVariableFromDisk('commonList')

		os.system("rm "+returnPathStruc2vec()+"/../pickles/distances-r-*.pickle")
		os.system("rm "+returnPathStruc2vec()+"/../pickles/distances-q-*.pickle")

		common_list_inverse = {}
		common_list_0 = {k:v[0][0] for k,v in commonList.items()}
		for k,v in common_list_0.items():
			if v not in common_list_inverse:
				common_list_inverse[v] = [k]
			else:
				common_list_inverse[v].append(k)
		ordered_common_list = sorted(common_list_inverse.keys())

		with ProcessPoolExecutor(max_workers = 1) as executor:


			logging.info("Split degree List...")
			part = 1
			for c in chunks:
				job = executor.submit(splitDegreeList,common_list_0, common_list_inverse, ordered_common_list,part,c,G,compactDegree)
				job.result()
				logging.info("degreeList {} completed.".format(part))
				part += 1

				
		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part))
				job = executor.submit(calc_distances, part, compactDegree = compactDegree)
				futures[job] = part
				part += 1
					
			logging.info("Receiving results...")
			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} completed.".format(r))

		part = 1
		for c in chunks:
			logging.info("Consolidating distances part {}...".format(part))
			distances_r = restoreVariableFromDisk('distances-r-'+str(part))
			distances_q = restoreVariableFromDisk('distances-q-'+str(part))
			preprocess_consolides_distances(distances_r)
			preprocess_consolides_distances(distances_q)
			saveVariableOnDisk(distances_r,'distances-r-'+str(part))
			saveVariableOnDisk(distances_q,'distances-q-'+str(part))
			part += 1
					
		return
	def consolide_distances(self):

		distances = {}

		parts = self.workers
		for part in range(1,parts + 1):
			d = restoreVariableFromDisk('distances-'+str(part))
			preprocess_consolides_distances(distances)
			distances.update(d)


		preprocess_consolides_distances(distances)
		saveVariableOnDisk(distances,'distances')


	def create_distances_network(self):

		with ProcessPoolExecutor(max_workers=1) as executor:
			job = executor.submit(generate_distances_network,self.workers, self.pcommonf)

			job.result()

		return

	def preprocess_parameters_random_walk(self):

		with ProcessPoolExecutor(max_workers=1) as executor:
			job = executor.submit(generate_parameters_random_walk,self.workers)

			job.result()

		return


	def simulate_walks(self,num_walks,walk_length):

		# for large graphs, it is serially executed, because of memory use.
		if(len(self.G) > 50000):

			with ProcessPoolExecutor(max_workers=1) as executor:
				job = executor.submit(generate_random_walks_large_graphs,num_walks,walk_length,self.workers,self.G.keys())

				job.result()

		else:

			with ProcessPoolExecutor(max_workers=1) as executor:
				job = executor.submit(generate_random_walks,num_walks,walk_length,self.workers,self.G.keys())

				job.result()


		return	





		

      	


