// Compile: g++ affinity_clustering.cpp -O2 -o affinity_clustering.exe

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <stdexcept>

using namespace std;

struct Edge;

struct Vertex
{
	Vertex(int _index) : index(_index), edges{}
		{}
	
	int index;
	vector<Edge> edges;
};

struct Edge
{
	Edge(Vertex* _endpoint, int _weight) : weight(_weight), endpoint(_endpoint)
		{}
	
	int weight;
	Vertex* endpoint;
};

void makeEdge(Vertex* v1, Vertex* v2, int weight)
{
	v1->edges.emplace_back(v2, weight);
	v2->edges.emplace_back(v1, weight);
}

Vertex* Lambda(const Vertex *v)
{
	if (v->edges.empty())
		throw runtime_error("Vertex " + to_string(v->index) + " heeft geen edges. Dikke programmeerfout.");
	
	const Edge* lowestEdge = &v->edges[0];
	
	for (size_t i = 1; i < v->edges.size(); ++i)
	{
		if (v->edges[i].weight < lowestEdge->weight)
			lowestEdge = &v->edges[i];
	}
	
	return lowestEdge->endpoint;
}

const Vertex* Map_First_Implementation(const Vertex* u)
{
	const Vertex* c = u, *v = u;
	set<const Vertex*> S{};
	
	while (S.find(v) == S.end())
	{
		S.insert(v);
		c = (c->index < v->index ? c : v);
		v = Lambda(v);
	}
	
	return c;
}

const Vertex* Map_Second_Implementation(const Vertex* u)
{
	const Vertex* c = u, *v = u;
	set<const Vertex*> S{};
	
	while (S.find(v) == S.end())
	{
		S.insert(v);
		v = Lambda(v);
	}
	
	return v;
}

const Vertex* Map_Third_Implementation(const Vertex* u)
{
	const Vertex* c = u, *v = u;
	set<const Vertex*> S{};
	
	while (S.find(v) == S.end())
	{
		S.insert(v);
		c = v;
		v = Lambda(v);
	}
	
	return (c->index < v->index ? c : v);
}


struct LeaderAlgorithm
{
	const Vertex* (*implementation)(const Vertex*);
	string name;
};

int main()
{
	#define v1 &graph[0]
	#define v2 &graph[1]
	#define v3 &graph[2]
	#define v4 &graph[3]
	#define v5 &graph[4]
	#define v6 &graph[5]
	#define v7 &graph[6]
	#define v8 &graph[7]
	#define v9 &graph[8]
	#define v10 &graph[9]
	
	vector<Vertex> graph{};
	
	for (int i = 0; i < 10; ++i)
		graph.emplace_back(i + 1);
	
	makeEdge(v10, v1, 2);
	makeEdge(v1, v2, 3);
	makeEdge(v2, v3, 2);
	makeEdge(v3, v4, 1);
	makeEdge(v4, v5, 2);
	makeEdge(v7, v5, 3);
	makeEdge(v8, v7, 5);
	makeEdge(v8, v6, 3);
	makeEdge(v9, v6, 2);
	makeEdge(v10, v9, 1);
	
	vector<LeaderAlgorithm> algorithms = { 
		{ Map_First_Implementation, string("Map-First-Implementation") },
		{ Map_Second_Implementation, string("Map-Second-Implementation") },
		{ Map_Third_Implementation, string("Map-Third-Implementation") }
	};
	
	for (const auto &algorithm : algorithms)
	{
		cout << algorithm.name << ':' << endl;
	
		for (const auto &v : graph)
		{
			cout << "Leader(v" << v.index << ')' << " = v" << algorithm.implementation(&v)->index << endl;
		}
		
		cout << endl;
	}	
	
	return 0;
}