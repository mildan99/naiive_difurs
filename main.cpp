#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
using std::cin,std::cout;

int main(){
int N;
cin>>N;
std::vector<int> v;
for(int i=0;i<N;++i){
   	int b;
   	cin>>b; 
	v.push_back(b);
}
if(!std::is_sorted(v.begin(),v.end())){
	cout<<-1;
}else{
std::set<int> s;
s.insert(v.begin(),v.end());
cout<<*std::max_element(s.begin(),s.end())-*std::min_element(s.begin(),s.end());
} 
}