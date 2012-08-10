/**
 * @file    GlobalRouter.cpp
 * @author  Ulrich Kemloh <kemlohulrich@gmail.com>
 * @version 0.1
 * Created on: Dec 15, 2010
 * Copyright (C) <2009-2011>
 *
 * @section LICENSE
 * This file is part of OpenPedSim.
 *
 * OpenPedSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * OpenPedSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OpenPedSim. If not, see <http://www.gnu.org/licenses/>.
 *
 * @section DESCRIPTION
 *
 *
 *
 */

#include "GlobalRouter.h"

#include "AccessPoint.h"
#include "Routing.h"
#include "../general/xmlParser.h"
#include "../geometry/Building.h"
#include "../pedestrian/Pedestrian.h"

#include <cfloat>
#include <fstream>


using namespace std;


GlobalRouter::GlobalRouter()
: Routing() {

	pAccessPoints.reserve(1500);
	pDistMatrix=NULL;
	pPathsMatrix=NULL;

}

GlobalRouter::~GlobalRouter() {

	if(pDistMatrix&&pPathsMatrix){
		const  int exitsCnt=GetAllGoals().size()/* +1*/;
		for(int p = 0; p < exitsCnt; ++p){
			delete [] pDistMatrix[p];
			delete [] pPathsMatrix[p];
		}

		delete  [] pDistMatrix;
		delete  [] pPathsMatrix;
	}

	while (!pAccessPoints.empty()){
		delete pAccessPoints.back();
		pAccessPoints.pop_back();
	}
	pAccessPoints.clear();
}

void GlobalRouter::Init(Building* building){

	Log->write("INFO:\tInit the Global Router Engine");
	pBuilding=building;

	//Load additional goals if necessary
	// performs basic initialization for Hermes
	//InitializeHermes();


	// initialize the network for the floydwarshall algo
	// initialize the distances matrix
	const  int exitsCnt=GetAllGoals().size()/* +1*/;

	pDistMatrix = new double*[exitsCnt];
	pPathsMatrix = new int*[exitsCnt];

	for(int i = 0; i < exitsCnt; ++i){
		pDistMatrix[i] = new double[exitsCnt];
		pPathsMatrix[i] = new int[exitsCnt];
	}
	//	initializing the values
	// all nodes are disconnected
	for(int p = 0; p < exitsCnt; ++p)
		for(int  r= 0; r < exitsCnt; ++r){
			pDistMatrix[p][r] = (r==p)?0.0:FLT_MAX;/*0.0*/;
			pPathsMatrix[p][r]=p;/*0.0*/;
		}


	// init the access points
	for (int door = 0; door < exitsCnt; door++) {

		Crossing *cross=GetAllGoals()[door];

		double x1=cross->GetPoint1().GetX();
		double y1=cross->GetPoint1().GetY();

		double x2=cross->GetPoint2().GetX();
		double y2=cross->GetPoint2().GetY();


		double center[2]={0 ,0};
		center[0] = 0.5*(x1+x2);
		center[1] = 0.5*(y1+y2);

		AccessPoint* ap = new AccessPoint(door, center);

		// save the connecting sub/rooms IDs
		int id1 = -1;
		if(cross->GetSubRoom1()) {
			id1=cross->GetSubRoom1()->GetUID();
		}

		int id2 = -1;
		if(cross->GetSubRoom2()) {
			id2=cross->GetSubRoom2()->GetUID();
		}

		ap->setConnectingRooms(id1,id2);

		//set the final destination
		if (cross->IsExit() && cross->IsOpen()){
			ap->setFinalDestination(true);
			//string caption=((Transition*)cross)->GetCaption();
			//printf("final destination found [%s] [%d] [%f,%f]: \n",caption.c_str(),door,center[0],center[1]);
			//cout <<"final destination found "<< door <<" "<< center[0] <<": "<<center[1]<<endl;
		} else if((id1==-1 ) && (id2==-1)){
			cout<<" a final destination outside the geometry was found"<<endl;
			ap->setFinalDestination(true);
		} else if (cross->GetRoom1()->GetCaption()=="outside"){
			ap->setFinalDestination(true);
		}

		pAccessPoints.push_back(ap);
	}

	// loop over the rooms
	// loop over the subrooms
	// get the transitions in the subrooms
	// and compute the distances


	for (int i = 0; i < pBuilding->GetAnzRooms(); i++) {
		Room* room = pBuilding->GetRoom(i);

		for (int j = 0; j < room->GetAnzSubRooms(); j++) {

			SubRoom* sub = room->GetSubRoom(j);
			const vector<int>& exitsInSubroom = sub->GetAllGoalIDs();

			// take one access point in that room and connect to all others

			for(unsigned int k=0;k< exitsInSubroom.size();k++){

				int from_door=exitsInSubroom[k];
				AccessPoint* from_AP=pAccessPoints[from_door];
				if(from_AP->GetID()!=from_door){
					Log->write(" an AP was not correct assigned");
					exit(EXIT_FAILURE);
				}

				Crossing* from_crossing=GetAllGoals()[from_door];

				if(from_crossing->IsOpen()==false){
					//string caption=((Transition*)from_crossing)->GetCaption();
					//Log->write("INFO:\t Transition closed : "+caption);
					continue;
				}

				// final destination are not connected
				if(from_AP->isFinalDestination()) continue;

				for(unsigned int l=0;l<exitsInSubroom.size();l++){
					// the entry is already 0 in the diagonal of the matrix
					int to_door=exitsInSubroom[l];
					Crossing* to_crossing=GetAllGoals()[to_door];

					//avoid connecting to myself
					if(from_door==to_door) continue;

					//skip closed doors
					if((to_crossing->IsOpen()==false)){
						continue;
					}

					if(from_crossing->IsOpen()==false){
						string caption=from_crossing->GetCaption();
						Log->write("INFO:\t Transition closed : "+caption);
						continue;
					}
					//workaround. Don't connect crossing from the tribune.
					// This will avoid getting from one rang into another

					if(Connectable(sub, from_door, to_door)==false) continue;

					if(CanSeeEachOther(from_crossing,to_crossing)==false) continue;

					// segment connecting the two APs/goals
					const Point& p1 = (from_crossing->GetPoint1()+from_crossing->GetPoint2())*0.5;
					const Point& p2 = (to_crossing->GetPoint1()+to_crossing->GetPoint2())*0.5;
					Line segment = Line(p1,p2);

					// check if this in intersected by any other connections/walls/doors/trans/cross in the room
					bool isVisible=true;

					//first walls
					const vector<Wall>& walls= sub->GetAllWalls();

					for(unsigned int b=0;b<walls.size();b++){
						if(segment.IntersectionWith(walls[b])==true) {
							isVisible=false;
							break;
						}
					}
					if(isVisible==false) continue;

					// then all goals
					for(unsigned int g=0;g<exitsInSubroom.size();g++){
						int gID=exitsInSubroom[g];
						// skip the concerned exits door and d
						if ((to_door==gID) || (from_door==gID)) continue;
						if(segment.IntersectionWith(*GetAllGoals()[gID])==true) {
							isVisible=false;
							break;
						}
					}
					if(isVisible==false) continue;

					double weight=1.0;

					// now  the final check
					// when a room is full with smoke pedes only go outside this room
					// not inside --> increasing the weight
					// this is only valid for transition
					if(to_crossing->IsTransition() && (from_crossing->IsTransition()==false)){
						//find the other room
						int room_id=from_crossing->GetRoom1()->GetRoomID();
						Room* other_room= ((Transition*)to_crossing)->GetOtherRoom(room_id);

						if(other_room && (other_room->GetRoomState()==ROOM_SMOKED)){
							// if all other doors are closed, they will be force to enter though
							weight=500;
							//cout<<" one smoked room found"<<endl;
						}
					}

					pDistMatrix[from_door][to_door]=weight*(p1-p2).Norm();

					from_AP->AddConnectingAP(pAccessPoints[to_door]);

					// in the case you want indirected graph
					// pDistMatrix[d][door]=pDistMatrix[l][k];
				}
			}
		}
	}

	// Handle the extra final destination which are not in any rooms
	// check which of the actual final destinations are connected to the extra

	for( map<int, int>::iterator it = pMapIdToFinalDestination.begin(); it != pMapIdToFinalDestination.end(); it++) {

		int to_door=it->first;
		AccessPoint* to_AP=pAccessPoints[to_door];
		printf("checking final [%d] aka [%s]\n",to_door,GetAllGoals()[to_door]->GetCaption().c_str());

		for(unsigned int i=0;i<pAccessPoints.size();i++){
			AccessPoint* from_AP=pAccessPoints[i];
			int from_door=from_AP->GetID();
			if(to_door==from_door) continue;

			// connect only the final exits to the virtual final destinations
			if(from_AP->isFinalDestination()==false) continue;

			// connect only open final exits
			//if(GetAllGoals()[from_door]->IsOpen()==false) continue;


			string to_room_caption=GetAllGoals()[to_door]->GetCaption();
			string from_room_caption="room_"+GetAllGoals()[from_door]->GetRoom1()->GetCaption();

			if(from_room_caption==to_room_caption){
				double dist=from_AP->GetDistanceTo(to_AP);
				dist=GetAllGoals()[to_door]->DistTo(from_AP->GetCentre());
				//cout<<"dist: "<<dist<<endl;
				//cout<<"from: "<<from_room_caption<<endl;
				//cout<<"to: "<<to_room_caption<<endl<<endl;
				//printf("from [%d] --> [%d]\n",from_door,to_door);
				//they are all situated at the same distance cuz they are virtual
				if(dist<18.0)
					pDistMatrix[from_door][to_door]=1.0;
			}

		}
	}
	//exit(0);
	//run the floyd warshall algorithm
	FloydWarshall();

	// set the configuration for reaching the outside
	// set the distances to all final APs
	for(unsigned int i=0;i<pAccessPoints.size();i++)
	{
		AccessPoint* from_AP=pAccessPoints[i];
		int from_door=from_AP->GetID();

		double tmpMinDist=FLT_MAX;
		int tmpFinalGlobalNearestID=from_door;

		for(unsigned int j=0;j<pAccessPoints.size();j++){
			AccessPoint* to_AP=pAccessPoints[j];
			int to_door=to_AP->GetID();
			if(from_door==to_door) continue;

			if(to_AP->isFinalDestination()){
				//cout <<" checking final destination: "<< pAccessPoints[j]->GetID()<<endl;
				double dist=pDistMatrix[from_door][to_door];
				if(dist<tmpMinDist){
					tmpFinalGlobalNearestID=to_door;
					tmpMinDist=dist;
				}
			}
		}

		// in the case it is the final APs
		if(tmpFinalGlobalNearestID==from_door) tmpMinDist=0;

		if(tmpMinDist==FLT_MAX){
			char tmp[CLENGTH];
			sprintf(tmp,"ERROR: GlobalRouter: There is no path from hline/crossing/transition [ %d ] to the outside\n",from_door);
			Log->write(tmp);
			exit(EXIT_FAILURE);
		}

		// set the distance to the final destination ( OUT )
		from_AP->AddFinalDestination(FINAL_DEST_OUT,tmpMinDist);

		// set the intermediate path to global final destination
		GetPath(from_door,tmpFinalGlobalNearestID);

		if (pTmpPedPath.size()>=2){
			//pAccessPoints[i]->setGlobalNearestAP(pTmpPedPath[1]);
			//from_AP->AddIntermediateDest(FINAL_DEST_OUT, pTmpPedPath[1]);
			from_AP->AddTransitAPsTo(FINAL_DEST_OUT, pAccessPoints[pTmpPedPath[1]]);
		}else{
			if((from_AP->isFinalDestination()==false) && (GetAllGoals()[from_door]->IsOpen())){
				char tmp[CLENGTH];
				const char* caption=GetAllGoals()[from_door]->GetRoom1()->GetCaption().c_str();
				sprintf(tmp,"ERROR: GlobalRouter: hline/crossing/transition [ %d ] in room [%s] is out of visibility range \n",from_door,caption);
				Log->write(tmp);
				exit(EXIT_FAILURE);
			}
		}
		pTmpPedPath.clear();
	}

	// set the configuration to reach alternatives (train, parking lot)
	// set the distances to alternative destinations

	for( map<int, int>::iterator it = pMapIdToFinalDestination.begin(); it != pMapIdToFinalDestination.end(); it++) {

		int to_door=it->first;
		int to_UID=it->second;

		// was already set
		if(to_UID==FINAL_DEST_OUT) continue;

		for(unsigned int i=0;i<pAccessPoints.size();i++){

			// set the distance
			AccessPoint* from_AP=pAccessPoints[i];
			int from_door=from_AP->GetID();

			// if enable, you will have problems if the investigated AP is a final destination
			//if(to_door==from_door) continue;

			//uncomment this if you want infinite as distance to unreachable destinations
			double dist=pDistMatrix[from_door][to_door];
			from_AP->AddFinalDestination(to_UID,dist);

			// set the intermediate path
			// set the intermediate path to global final destination
			GetPath(from_door,to_door);
			if (pTmpPedPath.size()>=2){
				//double dist=pDistMatrix[from_door][to_door];
				//from_AP->AddFinalDestination(to_UID,dist);
				//from_AP->AddIntermediateDest(to_UID,pTmpPedPath[1]);
				from_AP->AddTransitAPsTo(to_UID, pAccessPoints[pTmpPedPath[1]]);

			}else{
				if((from_AP->isFinalDestination()==false) && (GetAllGoals()[from_door]->IsOpen())){

					//we are still have some issues in the room 150
					string room_caption=GetAllGoals()[from_door]->GetRoom1()->GetCaption();
					if(room_caption=="150") continue;

					char tmp[CLENGTH];
					sprintf(tmp,"ERROR: GlobalRouter: hline/crossing/transition [ %d ] is out of visibility range 2\n",from_door);
					Log->write(tmp);
					sprintf(tmp,"ERROR: GlobalRouter: No path  from [ %d ] to [%d] in room [%s] \n",from_door,it->second,room_caption.c_str());
					Log->write(tmp);
					exit(EXIT_FAILURE);
				}
			}
			pTmpPedPath.clear();
		}
	}

	//create a complete navigation graph
	//LoadNavigationGraph("./Inputfiles/120531_navigation_graph_arena.xml");


	//dumping the complete system
	//DumpAccessPoints(825);
	//DumpAccessPoints(826);

	//DumpAccessPoints(1290);
	//DumpAccessPoints(1291);
	//DumpAccessPoints(1293);
	//DumpAccessPoints(1357);

	//WriteGraphGV("routing_graph.gv");
	//exit(0);

	Log->write("INFO:\tDone with the Global Router Engine!");
}


void GlobalRouter::AddFinalDestination(int UID, const Line& line){
	int id = GetAllGoals().size();

	Crossing* c = new Crossing();
	c->SetPoint1(line.GetPoint1());
	c->SetPoint2(line.GetPoint2());
	c->SetIndex(id);
	pMapIdToFinalDestination[id]=UID;

}

void GlobalRouter::InitializeHermesFinalDestinations(){

	// parking TOP
	if(pBuilding->GetTransition("AR0700279")->IsOpen()){
		int id=pBuilding->GetTransition("AR0700279")->GetIndex();
		pMapIdToFinalDestination[id]=FINAL_DEST_PARKING_TOP;
		char tmp[CLENGTH];
		sprintf(tmp,"\tInfo: Extra Final Destination loaded : Parking Top , AR0700279 with ID = %d",id);
		Log->write(tmp);
	}
	// parking BOTTOM
	if(pBuilding->GetTransition("AR0300058")->IsOpen()){
		int id=pBuilding->GetTransition("AR0300058")->GetIndex();
		pMapIdToFinalDestination[id]=FINAL_DEST_PARKING_BOTTOM;
		char tmp[CLENGTH];
		sprintf(tmp,"\tInfo: Extra Final Destination loaded : Parking Top , AR0300058 with ID = %d",id);
		Log->write(tmp);
	}
	// room_010
	if(pBuilding->GetGoal("room_010")->IsOpen()){
		int id=pBuilding->GetGoal("room_010")->GetIndex();
		pMapIdToFinalDestination[id]=FINAL_DEST_ROOM_010;
		char tmp[CLENGTH];
		sprintf(tmp,"\tInfo: Extra Final Destination loaded : room_010 with ID = %d",id);
		Log->write(tmp);
	}
	// room_020
	if(pBuilding->GetGoal("room_020")->IsOpen()){
		int id=pBuilding->GetGoal("room_020")->GetIndex();
		pMapIdToFinalDestination[id]=FINAL_DEST_ROOM_020;
		char tmp[CLENGTH];
		sprintf(tmp,"\tInfo: Extra Final Destination loaded : room_020 with ID = %d",id);
		Log->write(tmp);
	}
	// room_030
	if(pBuilding->GetGoal("room_030")->IsOpen()){
		int id=pBuilding->GetGoal("room_030")->GetIndex();
		pMapIdToFinalDestination[id]=FINAL_DEST_ROOM_030;
		char tmp[CLENGTH];
		sprintf(tmp,"\tInfo: Extra Final Destination loaded : room_030 with ID = %d",id);
		Log->write(tmp);
	}
	// room_040
	if(pBuilding->GetGoal("room_040")->IsOpen()){
		int id=pBuilding->GetGoal("room_040")->GetIndex();
		pMapIdToFinalDestination[id]=FINAL_DEST_ROOM_040;
		char tmp[CLENGTH];
		sprintf(tmp,"\tInfo: Extra Final Destination loaded : room_040 with ID = %d",id);
		Log->write(tmp);
	}
	// room_050
	if(pBuilding->GetGoal("room_050")->IsOpen()){
		int id=pBuilding->GetGoal("room_050")->GetIndex();
		pMapIdToFinalDestination[id]=FINAL_DEST_ROOM_050;
		char tmp[CLENGTH];
		sprintf(tmp,"\tInfo: Extra Final Destination loaded : room_050 with ID = %d",id);
		Log->write(tmp);
	}

	//train
	// the train station is the closest to room 020
	if(pBuilding->GetGoal("room_020")->IsOpen()){
		int id=pBuilding->GetGoal("room_020")->GetIndex();
		pMapIdToFinalDestination[id]=FINAL_DEST_ROOM_020;
		char tmp[CLENGTH];
		sprintf(tmp,"\tInfo: Extra Final Destination loaded TRAIN_STATION: room_020 with ID = %d",id);
		Log->write(tmp);
	}
}


void GlobalRouter::GetPath (int i, int j) {
	if (pDistMatrix[i][j]==FLT_MAX) return;
	if (i!=j)
		GetPath(i,pPathsMatrix[i][j]);
	pTmpPedPath.push_back(j);
	//printf("--%d--",j);
}

/*
	floyd_warshall()

	after calling this function dist[i][j] will the the minimum distance
	between i and j if it exists (i.e. if there's a path between i and j)
	or 0, otherwise
 */
void GlobalRouter::FloydWarshall() {
	//	int i, j, k;
	const int n=GetAllGoals().size()/* +1*/;

	for (int k=0;k<n;k++)
		for (int i=0;i<n;i++)
			for (int j=0;j<n;j++)
				if (pDistMatrix[i][k] + pDistMatrix[k][j] < pDistMatrix[i][j]) {
					pDistMatrix[i][j] = pDistMatrix[i][k]+pDistMatrix[k][j];
					pPathsMatrix[i][j] = pPathsMatrix[k][j];
				}
	return;

}

void GlobalRouter::DumpAccessPoints(int p){

	if(p!=-1){
		pAccessPoints.at(p)->Dump();
	}
	else
	{
		for(unsigned int i=0;i<pAccessPoints.size();i++){
			pAccessPoints[i]->Dump();
		}
	}
}


void GlobalRouter::InitializeHermes(){
	LoadAdditionalGoals("./Inputfiles/110818_arena.graph.xml");
	InitializeHermesFinalDestinations();
	CheckInconsistencies();

}

void GlobalRouter::LoadAdditionalGoals(string fileName){

	//return;
	Log->write("\tINFO: Loading additional goals from "+fileName);
	XMLNode xMainNode=XMLNode::openFileHelper(fileName.c_str(),"graph");

	XMLNode verticesList=xMainNode.getChildNode("Hlines");

	int verticesCount=verticesList.nChildNode("hline");

	int id=GetAllGoals().size();

	for (int i = 0; i < verticesCount; i++) {

		XMLNode vertex=verticesList.getChildNode("hline",i);

		int nodeID=atof(vertex.getAttribute("index"));
		double xPos1=atof(vertex.getAttribute("startX"));
		double yPos1=atof(vertex.getAttribute("startY"));
		double xPos2=atof(vertex.getAttribute("endX"));
		double yPos2=atof(vertex.getAttribute("endY"));
		const char* caption=vertex.getAttribute("caption");


		Point p1=Point(xPos1,yPos1);
		Point p2=Point(xPos2,yPos2);

		Crossing* c = new Crossing();
		c->SetPoint1(p1);
		c->SetPoint2(p2);
		c->SetIndex(id);
		c->SetSubRoom1(NULL);
		c->SetSubRoom2(NULL);
		c->SetRoom1(NULL);
		if(caption) c->SetCaption(caption);


		bool assigned=false;

		//look for the room/subroom
		for (int r = 0; r < pBuilding->GetAnzRooms(); r++) {
			Room* room = pBuilding->GetRoom(r);

			for (int j = 0; j < room->GetAnzSubRooms(); j++) {

				SubRoom* sub = room->GetSubRoom(j);
				if( (sub->IsInSubRoom(p1)) && (sub->IsInSubRoom(p2))){
					sub->AddGoalID(c->GetIndex());
					c->SetSubRoom1(sub);
					c->SetSubRoom2(sub);
					c->SetRoom1(room);
					assigned=true;
					break;
				}
			}
			if(assigned) break;
		}

		if(assigned){
			AddGoal(c);
			char tmp[CLENGTH];
			sprintf(tmp, "Additional Crossing with id %d [node=%d]  and caption %s loaded in room %s \n",id,nodeID,caption,c->GetRoom1()->GetCaption().c_str());
			Log->write(tmp);
			id++; // to enable get by index from the vector
		}
		else
		{
			char tmp[CLENGTH];
			sprintf(tmp,"The additional Crossing %d could not be assigned a room, it wont be added\n",nodeID);
			Log->write(tmp);
			//AddGoal(c);
			//id++;
			//getc(stdin);
			delete c;
		}
	}

	Log->write("\tINFO: Done !");
	//exit(0);
	//XMLNode xNode=xMainNode.getChildNode("Kantenliste");
}

void GlobalRouter::LoadNavigationGraph(string fileName){
	Log->write("\tINFO: Loading navigation graph from "+fileName);
	XMLNode xMainNode=XMLNode::openFileHelper(fileName.c_str(),"navigation_graph");

	int nav_graphs_cnt=xMainNode.nChildNode("graph");

	// first reset all access Point which are to be updated in the next steps
	for (int i = 0; i < nav_graphs_cnt; i++) {
			XMLNode nav_graph=xMainNode.getChildNode("graph",i);
			string final_dest=nav_graph.getAttribute("final_dest");
			int UID=FINAL_DEST_OUT;
			if (final_dest=="OUT") UID=FINAL_DEST_OUT;
			if (final_dest=="HRI010") UID=FINAL_DEST_ROOM_010;
			if (final_dest=="HRI020") UID=FINAL_DEST_ROOM_020;
			if (final_dest=="HRI030") UID=FINAL_DEST_ROOM_030;
			if (final_dest=="HRI040") UID=FINAL_DEST_ROOM_040;
			if (final_dest=="HRI050") UID=FINAL_DEST_ROOM_050;
			if (final_dest=="PNorth") UID=FINAL_DEST_PARKING_TOP;
			if (final_dest=="PWest" ) UID=FINAL_DEST_PARKING_BOTTOM;

			//get the different paths
			int paths_cnt=nav_graph.nChildNode("path");

			for (int j = 0; j < paths_cnt; j++) {
				XMLNode path=nav_graph.getChildNode("path",j);
				//only the first node is important
				int start=atoi(path.getAttribute("start"));
				pAccessPoints[start]->Reset(UID);
			}
		}

	// loop over the different nav graphs
	for (int i = 0; i < nav_graphs_cnt; i++) {
		XMLNode nav_graph=xMainNode.getChildNode("graph",i);

		string final_dest=nav_graph.getAttribute("final_dest");
		int UID=FINAL_DEST_OUT;
		if (final_dest=="OUT") UID=FINAL_DEST_OUT;
		if (final_dest=="HRI010") UID=FINAL_DEST_ROOM_010;
		if (final_dest=="HRI020") UID=FINAL_DEST_ROOM_020;
		if (final_dest=="HRI030") UID=FINAL_DEST_ROOM_030;
		if (final_dest=="HRI040") UID=FINAL_DEST_ROOM_040;
		if (final_dest=="HRI050") UID=FINAL_DEST_ROOM_050;
		if (final_dest=="PNorth") UID=FINAL_DEST_PARKING_TOP;
		if (final_dest=="PWest" ) UID=FINAL_DEST_PARKING_BOTTOM;


		//get the different paths
		int paths_cnt=nav_graph.nChildNode("path");
		cout<<"path count: "<<paths_cnt<<endl;

		for (int j = 0; j < paths_cnt; j++) {
			XMLNode path=nav_graph.getChildNode("path",j);
			int start=atoi(path.getAttribute("start"));
			int end=atoi(path.getAttribute("end"));
			pAccessPoints[start]->AddTransitAPsTo(UID,pAccessPoints[end]);
		}
		cout<<"final_dest= " << final_dest<<endl;
	}

	cout<<"done loading graph"<<endl;
	vector<string> rooms;
	rooms.push_back("010");
	//rooms.push_back("020");
	//rooms.push_back("030");
	//rooms.push_back("040");
	//rooms.push_back("050");
	//WriteGraphGV("routing_graph.gv",FINAL_DEST_PARKING_BOTTOM,rooms);
	//WriteGraphGV("routing_graph.gv",FINAL_DEST_OUT,rooms);
	//exit(0);
}

int GlobalRouter::FindExit(Pedestrian* ped){

	int nextDestination=ped->GetNextDestination();

	//int pedToLog=1;
	//if(ped->GetPedIndex()==5 && (nextDestination==553)){
	//if(ped->GetPedIndex()==1 && (nextDestination!=56)&& (ped->GetSubRoomID()==46)){
	//	if(ped->GetPedIndex()==-30){
	//		cout<<"---------------"<<endl;
	//		cout <<" pedID: "<<ped->GetPedIndex()<<endl;
	//		cout <<" roomID: "<<" [ "<< ped->GetRoomID()<<" "<<ped->GetSubRoomID()<<" ]"<<endl;
	//		cout <<" next: "<<nextDestination<<endl;
	//		cout<<"---------------"<<endl;
	//	}
	//get the room and find the corresponding node

	SubRoom* sub = pBuilding->GetRoom(ped->GetRoomID())->GetSubRoom(ped->GetSubRoomID());

	// get the opened exits
	int bestAPsID=-1;
	double minDist=FLT_MAX;

	const vector<int>& accessPointsInSubRoom =sub->GetAllGoalIDs();

	for(unsigned int i=0;i<accessPointsInSubRoom.size();i++){

		int apID=accessPointsInSubRoom[i];

		const Point& pt3=ped->GetPos();
		double distToExit=GetAllGoals()[apID]->DistTo(pt3);

		double tolerance=EPS_AP_DIST;
		if(GetAllGoals()[apID]->GetSubRoom1()==GetAllGoals()[apID]->GetSubRoom2()){
			tolerance=0.01;
		}
		if(distToExit >tolerance) continue;

		//one AP is near actualize destination:

		nextDestination=pAccessPoints[apID]->GetNearestTransitAPTO(ped->GetFinalDestination());

		//special case for rooms010/030
		if((pAccessPoints[apID]->GetID()==1178) || (pAccessPoints[apID]->GetID()==1189)) {
			const vector<AccessPoint*>& aps = pAccessPoints[apID]->GetTransitAPsTo(ped->GetFinalDestination());
			double distMin=FLT_MAX;
			int nearest=-1;
			for(unsigned int ap=0;ap<aps.size();ap++){
				double dist1=(aps[ap]->GetCentre()-ped->GetPos()).NormSquare();
				if(dist1<distMin){
					distMin=dist1;
					nearest=aps[ap]->GetID();
				}
			}
			nextDestination= nearest;
		}



		if(nextDestination==-1){ // we are almost at the exit
			nextDestination=apID;
			//cout<<"One pedestrian is right near to the exit: "<<apID<<endl;
		}

		//check that the next destination is in the actual room of the pedestrian
		if(pAccessPoints[nextDestination]->isInRange(ped->GetUniqueRoomID())==false) {
			//return the last destination if defined
			int previousDestination=ped->GetNextDestination();

			//we are still somewhere in the initialization phase
			if(previousDestination==-1){
				nextDestination=apID;
				ped->SetExitIndex(nextDestination);
				ped->SetExitLine(GetAllGoals()[nextDestination]);
				ped->SetSmoothTurning(true);
				//				if(ped->GetPedIndex()==pedToLog){
				//					cout<<"called1: "<< nextDestination<<endl;
				//					//cout<<"Raum/Subroom: "<<ped->GetRoomID()<<" / "<<ped->GetSubRoomID()<<endl;
				//				}

				return nextDestination;
			}
			else // we are still having a valid destination, don't change
			{
				//				if(ped->GetPedIndex()==pedToLog){
				//					cout<<"called2: "<<previousDestination<<endl;
				//					//cout<<"Raum/Subroom[in rout]: "<<ped->GetRoomID()<<" / "<<ped->GetSubRoomID()<<endl;
				//				}
				return previousDestination;
			}
		}
		else  // we have reached the new room
		{
			// only assigned if not already assigned
			// quite messed up code, sry
			//if(nextDestination!=ped->GetNextDestination()){
			ped->SetExitIndex(nextDestination);
			ped->SetExitLine(GetAllGoals()[nextDestination]);
			ped->SetSmoothTurning(true);
			return nextDestination;
		}

	}


	//any actual valid destinations?
	// if so keep it
	int next=ped->GetNextDestination();

	if(next!=-1) return next;

	//ich checke alles was ich sehen kann
	//dann entscheide ich  mich fuer die kurzeste Variante

	// in the case the previous attempt didnt work
	minDist=FLT_MAX;
	bestAPsID=-1;
	//int randomExit=0;

	for(unsigned int i=0;i<accessPointsInSubRoom.size();i++)
	{
		int apID=accessPointsInSubRoom[i];
		//randomExit=apID;
		int uniqueRoomID= ped->GetUniqueRoomID();
		if(pAccessPoints[apID]->isInRange(uniqueRoomID)==false) continue;

		//check if that exit is open.
		if(GetGoal(apID)->IsOpen()==false) continue;

		//avoid going into rang charakterized by size>10
		//20 is for room/subroom [070]0/0
		// provided there are more choices
		if ((GetGoal(apID)->Length()<0.53)&&(accessPointsInSubRoom.size()>10)) continue;

		//check if I can see that exit
		SubRoom* sub = pBuilding->GetRoom(ped->GetRoomID())->GetSubRoom(ped->GetSubRoomID());

		// segment connecting the two APs/goals
		const Point& p1 = (GetAllGoals()[apID]->GetPoint1()+GetAllGoals()[apID]->GetPoint2())*0.5;
		const Point& p2 = ped->GetPos();
		Line segment = Line(p1,p2);

		// check if this in intersected by any other connections/walls/doors/trans/cross in the room
		bool isVisible=true;

		//first walls
		const vector<Wall>& walls= sub->GetAllWalls();

		for(unsigned int b=0;b<walls.size();b++){
			if(segment.IntersectionWith(walls[b])==true) {
				isVisible=false;
				//cout<<"failed: walls "<<b<<" in subroom " << sub->GetSubRoomID()<<endl;
				break;
			}
		}
		if(isVisible==false) continue;

		// then all goals
		for(int g=0;g<(int)accessPointsInSubRoom.size();g++){
			int gID=accessPointsInSubRoom[g];
			if(gID==apID) continue;
			// skip the concerned exits door and d
			if(segment.IntersectionWith(*GetAllGoals()[gID])==true) {
				isVisible=false;
				//cout<<"failed: goal "<<gID<<endl;
				break;
			}
		}
		if(isVisible==false) continue;
		double x = ped->GetPos().GetX();
		double y = ped->GetPos().GetY();
		double dist=pAccessPoints[apID]->GetDistanceTo(ped->GetFinalDestination())+pAccessPoints[apID]->distanceTo(x,y);

		if(dist<minDist){
			bestAPsID=pAccessPoints[apID]->GetID();
			minDist=dist;
			//cout<<" best found: " <<apID<<" " <<bestAPsID<<endl;
		}
	}

	if (bestAPsID==-1){
		//will be deleted
		//FIXME
		ped->SetExitLine(GetAllGoals()[0]);
		return -1;

		char tmp[CLENGTH];
		const char* caption=pBuilding->GetRoom(ped->GetRoomID())->GetCaption().c_str();
		sprintf(tmp,"WARNING: GlobalRouter: best AP could not be identified for pedestrian %d in room/subroom [%s] %d/%d \n",ped->GetPedIndex(), caption, ped->GetRoomID(),ped->GetSubRoomID());
		Log->write(tmp);
		Log->write("WARNING: GlobalRouter: There are no exit in the sight range");
		//bestAPsID=randomExit;
		bestAPsID=GetBestDefaultRandomExit(ped);
		//return ped->GetNextDestination();
		//cout<<"sub="<<sub->GetAnzGoalIDs()<<endl;
		//cout<<"sub="<<bestAPsID<<endl;
		sprintf(tmp,"WARNING: GlobalRouter: I am choosing a random one [ %d ]",bestAPsID);
		Log->write(tmp);
		//exit(0);
	}

	nextDestination= bestAPsID;
	ped->SetExitIndex(nextDestination);
	ped->SetSmoothTurning(true);
	ped->SetExitLine(GetAllGoals()[nextDestination]);

	return nextDestination;
}

int GlobalRouter::GetBestDefaultRandomExit(Pedestrian* ped){

	SubRoom* sub = pBuilding->GetRoom(ped->GetRoomID())->GetSubRoom(ped->GetSubRoomID());

	// get the opened exits
	int bestAPsID=-1;
	double minDist=FLT_MAX;

	const vector<int>& accessPointsInSubRoom =sub->GetAllGoalIDs();

	for(unsigned int i=0;i<accessPointsInSubRoom.size();i++)
	{
		int apID=accessPointsInSubRoom[i];
		int uniqueRoomID= ped->GetUniqueRoomID();
		if(pAccessPoints[apID]->isInRange(uniqueRoomID)==false) continue;

		double x = ped->GetPos().GetX();
		double y = ped->GetPos().GetY();
		double dist=pAccessPoints[apID]->GetDistanceTo(ped->GetFinalDestination())+pAccessPoints[apID]->distanceTo(x,y);

		if(dist<minDist){
			bestAPsID=pAccessPoints[apID]->GetID();
			minDist=dist;
			//cout<<" best found: " <<apID<<" " <<bestAPsID<<endl;
		}
	}
	if(bestAPsID==-1){
		Log->write("ERROR:\t unrecoverable error in the routing class");
		Log->write("ERROR:\t the results/trajectories are now meaningless");
		return 0;
	}
	return bestAPsID;
}

bool GlobalRouter::CanSeeEachother(const Point&pt1, const Point& pt2){

	Line segment= Line(pt1,pt2);

	for (int i = 0; i < pBuilding->GetAnzRooms(); i++) {
		Room* room = pBuilding->GetRoom(i);

		for (int j = 0; j < room->GetAnzSubRooms(); j++) {
			SubRoom* sub = room->GetSubRoom(j);
			//  walls
			const vector<Wall>& walls= sub->GetAllWalls();

			for(unsigned int b=0;b<walls.size();b++){
				if(segment.IntersectionWith(walls[b])==true) {
					return false;
				}
			}
		}
	}

	// then all goals
	int nSize = GetAllGoals().size();
	for (int door = 0; door < nSize; door++) {
		Crossing *cross=GetAllGoals()[door];
		if(cross->GetRoom1()->GetCaption()=="outside") continue;
		if(segment.IntersectionWith(*cross)==true) {
			return false;
		}
	}

	return true;
}


//workaround. Don't connect crossing from the tribune.
// This will avoid getting from one rang into another
// both are crossings && both are <51cm && there are more than 2
bool GlobalRouter::Connectable(SubRoom* sub, int from, int to) const {

	//check if both are crossings or transitions
	Crossing* from_goal = GetAllGoals()[from];
	Crossing* to_goal = GetAllGoals()[to];

	//check if one of them are closed
	if((from_goal->IsOpen()==false) || (to_goal->IsOpen()==false)) return false;

	//check if hlines
	if (from_goal->GetSubRoom1() == from_goal->GetSubRoom2()) 	return true;
	if (to_goal->GetSubRoom1() == to_goal->GetSubRoom2()) 	return true;

	// we are not going in the rangs
	// provided it is possible
	if ((to_goal->Length()<=0.53)&&(from_goal->Length()<=0.53)){

		// avoid entering the rang
		if(sub->GetAllGoalIDs().size()>10) return false;
		//else return true;
	}

	//check if transition
	int id1 = -1;
	if(from_goal->GetSubRoom1()) {
		id1=from_goal->GetSubRoom1()->GetRoomID();
	}

	int id2 = -1;
	if(from_goal->GetSubRoom2()) {
		id2=from_goal->GetSubRoom2()->GetRoomID();
	}
	if(id1!=id2) return true;

	id1 = -1;
	if(to_goal->GetSubRoom1()) {
		id1=to_goal->GetSubRoom1()->GetRoomID();
	}

	id2 = -1;
	if(to_goal->GetSubRoom2()) {
		id2=to_goal->GetSubRoom2()->GetRoomID();
	}
	if(id1!=id2) return true;

	// at this this stage they should be both crossings in the same room
	// now check the size
	if (from_goal->Length()>0.53) return true;
	if (to_goal->Length()>0.53) return true;

	// another special case
	// necessary for room 100.
	// if the crossing are too close, connect
	if((from_goal->GetCentre()-to_goal->GetCentre()).Norm()<3.0 ) return true;

	return false;
}

bool GlobalRouter::CanSeeEachOther(Crossing* c1, Crossing* c2){

	//do they share at least one subroom?
	//find the common subroom,
	//return false if none
	SubRoom* sb1_a=c1->GetSubRoom1();
	SubRoom* sb1_b=c1->GetSubRoom2();

	SubRoom* sb2_a=c2->GetSubRoom1();
	SubRoom* sb2_b=c2->GetSubRoom2();

	SubRoom* sub=NULL;

	if((sb1_a!=NULL)&&(sb1_a==sb2_a)) sub=sb1_a;
	else if ((sb1_a!=NULL)&&(sb1_a==sb2_b))sub=sb1_a;
	else if ((sb1_b!=NULL)&&(sb1_b==sb2_a)) sub=sb1_b;
	else if ((sb1_b!=NULL)&&(sb1_b==sb2_b))sub=sb1_b;

	if(sub==NULL){
		//char tmp[CLENGTH];
		//sprintf(tmp,"no common subroom found for transitions [%d] and [%d]",
		//		c1->GetIndex(),c2->GetIndex());
		//Log->write(tmp);
		return false;
	}

	// segment connecting the two APs/goals
	const Point& p1 = (c1->GetPoint1()+c1->GetPoint2())*0.5;
	const Point& p2 = (c2->GetPoint1()+c2->GetPoint2())*0.5;
	Line segment = Line(p1,p2);

	// check if this in intersected by any other connections/walls/doors/trans/cross in the room

	//first walls
	const vector<Wall>& walls= sub->GetAllWalls();

	for(unsigned int b=0;b<walls.size();b++){
		if(segment.IntersectionWith(walls[b])==true) {
			return false;
		}
	}

	// also take into account other crossings/transitions
	const vector<int>& exitsInSubroom =sub->GetAllGoalIDs();

	int id1=c1->GetIndex();
	int id2=c2->GetIndex();
	// then all goals
	for(int g=0;g<(int)exitsInSubroom.size();g++){
		int gID=exitsInSubroom[g];
		// skip the concerned exits door and d
		if ((id1==gID) || (id2==gID)) continue;
		if(segment.IntersectionWith(*GetAllGoals()[exitsInSubroom[g]])==true) {
			return false;
		}
	}

	//last check in the case of a concav polygon
	// check if the middle of the connection line lies inside the subroom
	Point middle= (p1+p2)*0.5;
	bool isVisible= sub->IsInSubRoom(middle);

	if(isVisible==false){
		return false;
	}

	return true;
}

SubRoom* GlobalRouter::GetCommonSubRoom(Crossing* c1, Crossing* c2){
	SubRoom* sb11=c1->GetSubRoom1();
	SubRoom* sb12=c1->GetSubRoom2();
	SubRoom* sb21=c2->GetSubRoom1();
	SubRoom* sb22=c2->GetSubRoom2();

	if(sb11==sb21) return sb11;
	if(sb11==sb22) return sb11;
	if(sb12==sb21) return sb12;
	if(sb12==sb22) return sb12;

	return NULL;
}

void GlobalRouter::CheckInconsistencies(){

	Log->write("INFO: Checking all goals");
	// only check the hlines
	const vector<Crossing*>& goals=pBuilding->GetRouting()->GetAllGoals();

	for(unsigned int g=0; g<goals.size(); g++){

		Crossing * c1=goals[g];

		//that goal is located outside the complete geometry
		if(!c1->GetSubRoom1()){
			continue;
		}

		if(c1->GetSubRoom1()==c1->GetSubRoom2()){
			//cout<<"checking: "<<c1->GetIndex()<<endl;
			const Point& p1 = (c1->GetPoint1()+c1->GetPoint2())*0.5;
			if(c1->GetSubRoom1()->IsInSubRoom(p1)==false){
				SubRoom* sub = c1->GetSubRoom1();
				char tmp[CLENGTH];
				sprintf(tmp,"Hlines [%d] in room/subroom %s [%d / %d] failed. It is probably somewhere else",
						c1->GetIndex(),pBuilding->GetRoom(sub->GetRoomID())->GetCaption().c_str(),
						sub->GetRoomID(),sub->GetSubRoomID());
				Log->write(tmp);
				Log->write("I am trying to fix that.");

				bool fixed=false;
				// I only check that one room
				Room* room=c1->GetRoom1();
				for (int j = 0; j < room->GetAnzSubRooms(); j++) {
					SubRoom* sub1 = room->GetSubRoom(j);
					//set the new room if needed
					if (sub1->IsInSubRoom(p1)) {
						c1->SetSubRoom1(sub1);
						c1->SetSubRoom2(sub1);
						fixed=true;
						sprintf(tmp,"Fixed! relocated to room/subroom %s [%d / %d]",
								pBuilding->GetRoom(sub1->GetRoomID())->GetCaption().c_str(),
								sub1->GetRoomID(),sub1->GetSubRoomID());
						Log->write(tmp);

						//just dont forget this
						sub->RemoveGoalID(c1->GetIndex());
						sub1->AddGoalID(c1->GetIndex());
						break;
					}
				}

				if(fixed==false){
					Log->write("Could not fix...");
				}
			}
		}
	}
	Log->write("INFO: Checking all goals  done!");
}


void GlobalRouter::WriteGraphGV(string filename, int finalDestination, const vector<string> rooms_captions){
	ofstream graph_file (filename.c_str());
	if (graph_file.is_open()==false){
		Log->write("Unable to open file"+filename);
		return;
	}

	//header
	graph_file << "## Produced by OPS_GCFM"<<endl;
	//graph_file << "##comand: \" sfdp -Goverlap=prism -Gcharset=latin1"<<filename <<"| gvmap -e | neato -Ecolor=\"#55555522\" -n2 -Tpng > "<< filename<<".png \""<<endl;
	graph_file << "##Command to produce the output: \"neato -n -s -Tpng "<< filename<<" > "<< filename<<".png\""<<endl;
	graph_file << "digraph OPS_GCFM_ROUTING {"<<endl;
	graph_file << "overlap=scale;"<<endl;
	graph_file << "splines=false;"<<endl;
	graph_file << "fontsize=20;"<<endl;
	graph_file << "label=\"Graph generated by the routing engine for destination: "<< finalDestination<<"\"" <<endl;

	vector<int> rooms_ids=vector<int>();

	if(rooms_captions.empty()){
		// then all rooms should be printed
		for(int i=0; i<pBuilding->GetAnzRooms();i++){
			rooms_ids.push_back(i);
		}

	}else{
		for(unsigned int i=0; i<rooms_captions.size();i++){
			rooms_ids.push_back(pBuilding->GetRoom(rooms_captions[i])->GetRoomID());
		}
	}

	// all nodes
	for(unsigned int i=0;i<pAccessPoints.size();i++)
	{
		AccessPoint* from_AP=pAccessPoints[i];
		int from_door=from_AP->GetID();

		// check for valid room
		Crossing* cross=GetAllGoals()[from_door];
		int room_id=cross->GetRoom1()->GetRoomID();
		if(IsElementInVector(rooms_ids,room_id)==false) continue;


		double px=from_AP->GetCentre().GetX();
		double py=from_AP->GetCentre().GetY();
		//graph_file << from_door <<" [shape=ellipse, pos=\""<<px<<", "<<py<<" \"] ;"<<endl;
		//graph_file << from_door <<" [shape=ellipse, pos=\""<<px<<","<<py<<"\" ];"<<endl;

		//const vector<AccessPoint*>& from_aps = from_AP->GetConnectingAPs();
		const vector<AccessPoint*>& from_aps = from_AP->GetTransitAPsTo(finalDestination);

		if(from_aps.size()==0){

			if(from_AP->isFinalDestination()){
				graph_file << from_door <<" [pos=\""<<px<<", "<<py<<" \", style=filled, color=green,fontsize=5] ;"<<endl;
				//				graph_file << from_door <<" [width=\"0.41\", height=\"0.31\",fixedsize=false,pos=\""<<px<<", "<<py<<" \", style=filled, color=green,fontsize=4] ;"<<endl;
			}else{
				graph_file << from_door <<" [pos=\""<<px<<", "<<py<<" \", style=filled, color=red,fontsize=5] ;"<<endl;
				//				graph_file << from_door <<" [width=\"0.41\", height=\"0.31\",fixedsize=false,pos=\""<<px<<", "<<py<<" \", style=filled, color=red,fontsize=4] ;"<<endl;
			}
		}else{
			// check that all connecting aps are contained in the room_ids list
			// if not marked as sink.
			bool isSink=true;
			for(unsigned int j=0;j<from_aps.size();j++)
			{
				AccessPoint* to_AP=from_aps[j];
				int to_door=to_AP->GetID();

				Crossing* cross=GetAllGoals()[to_door];
				int room_id=cross->GetRoom1()->GetRoomID();
				if(IsElementInVector(rooms_ids,room_id)==true){
					isSink=false;
					break;
				}
			}

			if(isSink){
				//				graph_file << from_door <<" [width=\"0.3\", height=\"0.21\",fixedsize=false,pos=\""<<px<<", "<<py<<" \" ,style=filled, color=green, fontsize=4] ;"<<endl;
				graph_file << from_door <<" [pos=\""<<px<<", "<<py<<" \" ,style=filled, color=green, fontsize=5] ;"<<endl;
			}else{
				//				graph_file << from_door <<" [width=\"0.3\", height=\"0.231\",fixedsize=false, pos=\""<<px<<", "<<py<<" \", fontsize=4] ;"<<endl;
				graph_file << from_door <<" [pos=\""<<px<<", "<<py<<" \", style=filled, color=yellow, fontsize=5] ;"<<endl;
			}
		}

	}

	//connections
	for(unsigned int i=0;i<pAccessPoints.size();i++)
	{
		AccessPoint* from_AP=pAccessPoints[i];
		int from_door=from_AP->GetID();

		//const vector<AccessPoint*>& aps = from_AP->GetConnectingAPs();
		const vector<AccessPoint*>& aps = from_AP->GetTransitAPsTo(finalDestination);


		Crossing* cross=GetAllGoals()[from_door];
		int room_id=cross->GetRoom1()->GetRoomID();
		if(IsElementInVector(rooms_ids,room_id)==false) continue;

		for(unsigned int j=0;j<aps.size();j++)
		{
			AccessPoint* to_AP=aps[j];
			int to_door=to_AP->GetID();

			Crossing* cross=GetAllGoals()[to_door];
			int room_id=cross->GetRoom1()->GetRoomID();
			if(IsElementInVector(rooms_ids,room_id)==false) continue;

			graph_file << from_door <<" -> "<<to_door<<" [ label="<<from_AP->GetDistanceTo(to_AP) +to_AP->GetDistanceTo(finalDestination)<<", fontsize=10]; "<<endl;
		}

	}
	//graph_file << "node [shape=box];  gy2; yr2; rg2; gy1; yr1; rg1;"<<endl;
	//graph_file << "node [shape=circle,fixedsize=true,width=0.9];  green2; yellow2; red2; safe2; safe1; green1; yellow1; red1;"<<endl;

	//graph_file << "0 -> 1 ;"<<endl;

	graph_file << "}"<<endl;


	//done
	graph_file.close();
}