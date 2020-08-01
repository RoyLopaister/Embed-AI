#include<iostream>
#include "model.h"

using namespace std;

float model(float* in){
	float neuron[12]={0};
	for(int ii=0;ii<topology[0];ii++){
		neuron[ii]=in[ii];
		// cout<<"neuron["<<ii<<"]=in["<<ii<<"]"<<endl;
	}
	int begin=0;
	for(int layar=0;layar<sizeof(topology)/sizeof(topology[0])-1;layar++){
		// cout<<"node"<<begin+i+topology[layar]<<"="<<neuron[begin+i+topology[layar]]<<endl;
		
		int count=0;
		for(int in_node=begin;in_node<begin+topology[layar];in_node++){
			for(int nextlayar=0;nextlayar<topology[layar+1];nextlayar++){
				// cout<<"layar "<<layar<<" to layar "<<layar+1<<"\t"<<"("<<in_node<<", ";
				// cout<<begin+topology[layar]+nextlayar<<")";
				neuron[begin+topology[layar]+nextlayar]+=neuron[in_node]*weight[layar][count];
				// cout<<"\t\tneuron["<<begin+topology[layar]+nextlayar<<"] += neuron["<<in_node<<"] * w["<<layar<<"]["<<count<<"]="<<neuron[in_node]<<" * "<<weight[layar][count]<<" = "<<neuron[begin+topology[layar]+nextlayar]<<"\n";
				count++;
			}
		}
		for(int nextlayar=0;nextlayar<topology[layar+1];nextlayar++){
			// cout<<"\t\tneuron["<<begin+topology[layar]+nextlayar<<"] += neuron["<<begin+topology[layar]+nextlayar<<"] + b["<<begin+topology[layar]+nextlayar-1<<"]="<<neuron[begin+topology[layar]+nextlayar]<<" + "<<bias[begin+topology[layar]+nextlayar-1]<<" = ";
			neuron[begin+topology[layar]+nextlayar]+=bias[begin+topology[layar]+nextlayar-1];
			// cout<<neuron[begin+topology[layar]+nextlayar]<<"\n";
			neuron[begin+topology[layar]+nextlayar]=ReLu(neuron[begin+topology[layar]+nextlayar]);
		}
		// cout<<"----------------"<<endl;
		begin=topology[layar+1]*(layar)+topology[0];
	}
	return neuron[outnode-1];
}

int main(){
	float input[1]={-1};
	float a=AI.model(input);
	cout<<"\n\nInput value : "<<input[0]<<"\nPrediction Result : "<<a<<"\n\n"<<endl;
}
