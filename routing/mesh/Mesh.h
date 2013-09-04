/*
 * mesh.h
 *
 *  Created on: 21.08.2013
 *      Author: dominik
 */

#ifndef MESH_H_
#define MESH_H_

#include <fstream>
#include <vector>
#include "../../geometry/Point.h"
#include "../../geometry/NavLine.h"

class MeshEdge;
class MeshCell;
class MeshCellGroup;

class MeshData{
public:
	MeshData();
	~MeshData();
	MeshData(std::vector<Point*>,std::vector<MeshEdge*>,
			std::vector<MeshEdge*>,std::vector<MeshCellGroup*>);
	std::vector<Point*> Get_nodes(){return _mNodes;}
	std::vector<MeshEdge*> Get_edges(){return _mEdges;}
	std::vector<MeshEdge*> Get_outEdges(){return _mOutEdges;}
	std::vector<MeshCellGroup*> Get_cellGroups(){return _mCellGroups;}
	unsigned int Get_cellCount(){return _mCellCount;};

	MeshCell* GetCellAtPos(unsigned int tpos);

	MeshCell* FindCell(Point testp,int& cell_id);

private:
	std::vector<Point*> _mNodes;
	std::vector<MeshEdge*> _mEdges;
	std::vector<MeshEdge*> _mOutEdges;
	std::vector<MeshCellGroup*> _mCellGroups;
	unsigned int _mCellCount;

};

class MeshEdge:public NavLine{
public:
	MeshEdge(int,int,int,int,Point p1=Point(),Point p2=Point());//:Line(p1,p2);
	int Get_n1(){return _n1;};
	int Get_n2(){return _n2;};
	int Get_c1(){return _c1;};
	int Get_c2(){return _c2;};
	//friend std::istream& operator>>(std::istream& is, MeshEdge& mn);
private:
	int _n1; //ID of Node 1
	int _n2; //ID of Node 2
	int _c1; //ID of Cell 1
    int _c2; //ID of Cell 2
};

class MeshCell{
public:
	MeshCell(double,double,std::vector<int>,
			 double*,std::vector<int>,std::vector<int>,int);
	~MeshCell();
	//double get_midx(){return _midx;};
	//double get_midy(){return _midy;};
	Point Get_mid(){return _mid;};
	std::vector<int> Get_nodes(){return _node_id;};
	std::vector<int> Get_edges(){return _edge_id;};
	int Get_id(){return _tc_id;};
private:
	//double _midx;
	//double _midy;
	Point _mid;
	std::vector<int> _node_id;
	//double *_normvec;
	double _normvec[3];
	std::vector<int> _edge_id;
	std::vector<int> _wall_id;
	int _tc_id;//Cell ID unique for all cells in building
};

class MeshCellGroup{
public:
	MeshCellGroup(std::string,std::vector<MeshCell*>);
	~MeshCellGroup();
	std::vector<MeshCell*> Get_cells();
private:
    std::string _groupname;
    std::vector<MeshCell*> _cells;
};

unsigned int Calc_CellCount(std::vector<MeshCellGroup*> mcg);

#endif /* MESH_H_ */