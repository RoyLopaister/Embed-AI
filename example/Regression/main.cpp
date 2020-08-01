#include<iostream>
#include "EmbedAI.h"

using namespace std;


EmbedAI AI;

int main(){
	float input[1]={-1};
	float a=AI.model(input);
	cout<<"\n\nInput value : "<<input[0]<<"\nPrediction Result : "<<a<<"\n\n"<<endl;
}