import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
import collections
import pandas as pd
from collections import OrderedDict

class RWPara():
	__slots__ = ['epsilon', 'T', 'startVertex']

	def __init__(self, epsilon, T, startVertex):
		self.epsilon = epsilon
		self.T = T
		self.startVertex = startVertex

class Vertex:
	__slots__ = ['Neighbors', 'Weights', 'weightSum']

	def __init__(self, Neighbors, Weights, weightSum):
		self.Neighbors = Neighbors
		self.Weights = Weights
		self.weightSum = weightSum
	def setNeighbor(self, pos, j):
		self.Neighbors[pos] = j
	def setWeight(self, pos, j):
		self.Weights[pos] = j

class Graph:
	__slots__ = ['vertexList', 'cn', 'n', 'totalWeightSum', 'maxWeightSum']

	def __init__(self, vertexList, cn, n, totalWeightSum, maxWeightSum):
		self.vertexList = vertexList
		self.cn = cn
		self.n = n
		self.totalWeightSum = totalWeightSum
		self.maxWeightSum = maxWeightSum

def readData(filePath):
	global dataDictRow
	global dataDictCol
	i = 0

	data = pd.read_csv(filePath,sep=',',header=None).values
	data = np.row_stack(([[-1,-1],[-1,-1]],data)) # placeholder for node s and t
	data = np.column_stack((data,[0]*len(data)))
	print ('finished reading data')

	# integer encoded
	labelEncoder = LabelEncoder()
	valueDataRow = labelEncoder.fit_transform(data[2:,0])
	valueDataCol = labelEncoder.fit_transform(data[2:,1])
	dataDictRow = dict(zip(valueDataRow,data[2:,0]))
	dataDictCol = dict(zip(valueDataCol,data[2:,1]))
	data[2:,0] = valueDataRow
	data[2:,1] = valueDataCol

	print ('finished labelling')

	data = data.astype('float')

	# by column
	data = np.array(sorted(data,key=lambda x:x[1]))

	dataj = data[:,0:2].astype('int')
	DArr = collections.Counter(dataj[2:,1])

	cn = dataj[-1,1] + 1 # no. of column-oriented vertexes
	n = cn + max(dataj[:,0]) + 1 # no. of vertexes
	vertexList = [None]*n

	totalWeightSum = 0
	maxWeightSum = 0
	l = 2
	for i in DArr:
		wArr = [1]*(DArr[i]+2)

		weightSum = sum(wArr[2:])
		vertexList[i] = Vertex(dataj[l-2:l+DArr[i],0]+cn,wArr,weightSum)
		totalWeightSum += weightSum
		if weightSum>maxWeightSum: maxWeightSum=weightSum
		data[l:l+DArr[i],2] = wArr[2:]
		l += DArr[i]

	print ('finished setting col-data into Graph')

	# by row
	data = np.array(sorted(data,key=lambda x:x[0]))

	dataj = data[:,0:2].astype('int')

	DArr = collections.Counter(dataj[2:,0])

	l = 2
	for i in DArr:
		weightSum = sum(data[l:l+DArr[i],2])
		vertexList[i+cn] = Vertex(dataj[l-2:l+DArr[i],1].copy(), \
					data[l-2:l+DArr[i],2].copy(),weightSum)
		if weightSum>maxWeightSum: maxWeightSum=weightSum
		l += DArr[i]
	print ('finished setting row-data into Graph')

	return Graph(vertexList,cn,n,totalWeightSum*2,maxWeightSum)

# add node s and t
def reconstructGraph(G):
	tws = G.totalWeightSum
	m = G.maxWeightSum+1
	n = G.n
	vl = G.vertexList + [Vertex(np.array([-1]*n).astype('int'),[0]*n,0), \
		Vertex(np.array([-1]*n).astype('int'),[0]*n,0)]
	G.vertexList = vl
	for i in range(n):
		vl[n].setNeighbor(i,i)
		vl[n].setWeight(i,m)
		vl[n+1].setNeighbor(i,i)
		vl[n+1].setWeight(i,m-vl[i].weightSum)

		vl[i].setNeighbor(0,n)
		vl[i].setNeighbor(1,n+1)
		vl[i].setWeight(0,m)
		vl[i].setWeight(1,m-vl[i].weightSum)
		vl[i].weightSum = 2*m
	vl[n].weightSum = n*m
	vl[n+1].weightSum = n*m-tws
	G.n += 2
	G.totalWeightSum = 2*n*m
	print ('finished reconstructing Graph')

	return G

def randomWalk(G, RWP):
	vl = G.vertexList
	n = G.n
	sv = RWP.startVertex # startVertex
	epsilon = RWP.epsilon
	T = RWP.T
	
	key = [i for i in range(n)]
	zeroValue = [0]*n
	q2 = dict(zip(key,zeroValue))
	q2[sv] = 1

	for i in range(0,T):
		# # obtain maximum density subgraph
		# (bestDensity,bestSubgraph,q2) = calculateMaxDen(G,q)

		# one random walk loop
		q = dict(zip(q2.keys(),zeroValue))
		for j in q2.keys():
			qj = q2[j]
			if qj <= epsilon: continue
			q[j] += qj/2 # I/2*r
			nbIncC = qj/(2*vl[j].weightSum) # AD^(-1)/2*r
			for k in range(len(vl[j].Neighbors)):
				q[vl[j].Neighbors[k]] += vl[j].Weights[k]*nbIncC
		q2 = q
	print ('finished random walk')

	# obtain maximum density subgraph
	(bestDensity,bestSubgraph,q2) = calculateMaxDen(G,q)
	print ('finished maximum density calculation')

	return (bestDensity,bestSubgraph)

# calculate maximum density
def calculateMaxDen(G, q):
	vl = G.vertexList
	sv = G.n-2 # sourceVertex
	tv = G.n-1 # sinkVertex
	q = OrderedDict(sorted(q.items(), key=lambda x:x[1]/vl[x[0]].weightSum, reverse=True))
	
	D = OrderedDict()
	size = 0
	vol = 0
	cut = 0
	bestDensity = -1
	bestSubgraph = -1
	for i in q.keys():
		if 0 == q[i]: break
		if sv == i or tv == i: continue # "eliminate node s and t"
		D[i] = 0
		size += 1
		vol += vl[i].weightSum
		for j in range(len(vl[i].Neighbors)):
			if vl[i].Neighbors[j] in D.keys(): cut -= vl[i].Weights[j] # 'in D' here: 'in D.keys()'
			else: cut += vl[i].Weights[j]
		density = (vol-cut)/(2*size)

		if density > bestDensity:
			bestDensity = density
			bestSubgraph = size

	return (bestDensity,list(D)[0:bestSubgraph],q)

def decodeFinalSet(bestSubgraph):
	bestSubgraph = np.array(bestSubgraph)
	finalColSet = bestSubgraph[bestSubgraph<G.cn]
	finalRowSet = bestSubgraph[bestSubgraph>=G.cn]-G.cn
	for i in range(len(finalRowSet)): finalRowSet[i] = dataDictRow[finalRowSet[i]]
	for i in range(len(finalColSet)): finalColSet[i] = dataDictCol[finalColSet[i]]
	return (finalRowSet, finalColSet)

# Change the original r(no. rn) and c(no. cn) code to RW(no. n) code.
# Pay attention: rcCode is not subsequent (cuz from sparse matrix),
# while listCode is subsequent.
def rcCodeToListCode(rc, rcCode):
	if 'c' == rc:
		listCode = list(dataDictCol.keys())[list(dataDictCol.values()).index(rcCode)]
	else:
		listCode = list(dataDictRow.keys())[list(dataDictRow.values()).index(rcCode)]+G.cn
	return listCode

# ==================
# main
filePath = 'twitter.csv'
T = 10
# start_time = time.time()
G = readData(filePath)
# print ('readData: ',time.time() - start_time)
# start_time = time.time()
G = reconstructGraph(G)
# print ('reconstructGraph: ',time.time() - start_time)

# ==================
start_time = time.time()
bestDensity = -1
bestStartVertex = rcCodeToListCode('c', 15947185) # @tweetsmarter
(bestDensity,bestSubgraph) = randomWalk(G, RWPara(0,T,bestStartVertex))

print ('========')
# listCode to rcCode
if bestStartVertex in dataDictCol.keys():
	print ('bestStartVertex: c',dataDictCol[bestStartVertex])
else:
	print ('bestStartVertex: r',dataDictRow[bestStartVertex-G.cn])
print ('bestDensity: ',bestDensity)

(finalRowSet,finalColSet) = decodeFinalSet(bestSubgraph)
finalRowSet = list(finalRowSet)
finalColSet = list(finalColSet)
print ('finalRowSet: ',finalRowSet)
print ('finalColSet: ',finalColSet)
print ('finalRowSetNum: ',len(finalRowSet))
print ('finalColSetNum: ',len(finalColSet))
print ('G.cn: ',G.cn,'G.n: ',G.n)

print ('Random Walk: ',time.time() - start_time)